use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use yolocam::config::{Config, AUTO_MODEL_PATH};
use yolocam::database::{Database, HistoryFilter, StoredCounts};
use yolocam::detection::cleanup_screenshots;
use yolocam::inference::InferenceWorker;
use yolocam::rtsp::probe_ffmpeg;
use yolocam::state::{AiState, AiStatus, AppState};

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %error, "yolocam stopped due to an error");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("yolocam=info,ort=warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(true)
        .try_init();
}

async fn run() -> Result<(), String> {
    let config_path = parse_args()?;
    let config = Config::load(&config_path).map_err(|error| {
        format!(
            "cannot load {}: {}",
            config_path.display(),
            error.safe_message()
        )
    })?;

    let camera_url = config
        .public_rtsp_url()
        .unwrap_or_else(|_| "<invalid RTSP URL>".to_string());
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        camera = %config.camera.name,
        rtsp = %camera_url,
        interval_ms = config.yolo.interval_ms,
        threshold = config.yolo.confidence_threshold,
        model = %config.yolo.model_path,
        "starting yolocam"
    );
    if config.env_camera_locked {
        tracing::info!("RTSP credentials are managed by environment variables");
    }

    let screenshot_directory = config.screenshot_dir();
    if let Err(error) = std::fs::create_dir_all(&screenshot_directory) {
        tracing::warn!(
            directory = %screenshot_directory.display(),
            error = %error,
            "cannot create screenshot directory; screenshots will be unavailable"
        );
    }

    match probe_ffmpeg(&config.camera.ffmpeg_path).await {
        Ok(version) => tracing::info!(%version, "FFmpeg found"),
        Err(error) => tracing::warn!(
            error = %error,
            "FFmpeg is not available yet; the dashboard will report connection errors"
        ),
    }

    if config.yolo.model_path != AUTO_MODEL_PATH
        && !std::path::Path::new(&config.yolo.model_path).is_file()
    {
        tracing::warn!(
            model = %config.yolo.model_path,
            "YOLO model file not found; AI features will report an error until it is provided"
        );
    }

    let database_path = config.database_path();
    let database = Database::open(&database_path)
        .map_err(|error| format!("cannot open database {}: {error}", database_path.display()))?;
    let stored_counts = database.counts().await.unwrap_or_else(|error| {
        tracing::warn!(error = %error, "cannot read stored detection totals; starting at zero");
        StoredCounts::default()
    });
    let last_detection_at = match database
        .history(HistoryFilter {
            limit: 1,
            ..Default::default()
        })
        .await
    {
        Ok(records) => records.first().map(|record| record.timestamp.clone()),
        Err(error) => {
            tracing::warn!(error = %error, "cannot read the last detection timestamp");
            None
        }
    };
    tracing::info!(
        database = %database_path.display(),
        people_events = stored_counts.people,
        car_events = stored_counts.cars,
        "detection database ready"
    );

    let app = AppState::new(
        config.clone(),
        config_path,
        database,
        stored_counts,
        last_detection_at,
    );

    let bind_address = config.server.bind_address();
    let listener = match TcpListener::bind(&bind_address).await {
        Ok(listener) => listener,
        Err(error) => {
            let _ = app.database().shutdown().await;
            return Err(format!(
                "cannot bind dashboard server to {bind_address}: {error}"
            ));
        }
    };
    let local_address = listener
        .local_addr()
        .map(|address| address.to_string())
        .unwrap_or(bind_address.clone());

    let inference = match InferenceWorker::spawn(app.clone()) {
        Ok(worker) => Some(worker),
        Err(error) => {
            let message = format!("cannot start inference thread: {error}");
            tracing::error!(error = %message);
            app.set_ai_status(AiStatus {
                state: AiState::Error,
                message: Some(message),
                model_path: config.yolo.model_path.clone(),
                device: config.yolo.device,
                input_width: None,
                input_height: None,
                auto: None,
            });
            None
        }
    };

    let cleanup_task = spawn_screenshot_cleanup(app.clone());
    app.camera().start(app.clone()).await;

    let display_host = if config.server.host == "0.0.0.0" || config.server.host == "::" {
        "localhost".to_string()
    } else {
        config.server.host.clone()
    };
    tracing::info!(
        url = %format!("http://{display_host}:{}", config.server.port),
        bound = %local_address,
        "dashboard listening"
    );

    let router = yolocam::api::router(app.clone());
    let shutdown_token = app.shutdown_token();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown_token.cancelled().await })
            .await
    });

    let mut server = server;
    let mut server_error: Option<String> = None;
    let mut server_finished = false;
    tokio::select! {
        result = &mut server => {
            server_finished = true;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => server_error = Some(format!("dashboard server failed: {error}")),
                Err(error) => server_error = Some(format!("dashboard server task failed: {error}")),
            }
            tracing::info!("dashboard server stopped");
        }
        _ = wait_for_shutdown_signal() => {
            tracing::info!("shutdown signal received");
            app.request_shutdown();
        }
    }

    if !app.is_shutting_down() {
        app.request_shutdown();
    }

    // Give the HTTP server a short grace period to close WebSocket/MJPEG
    // connections, then release the remaining workers in dependency order.
    if !server_finished {
        match tokio::time::timeout(Duration::from_secs(10), &mut server).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(error))) => {
                tracing::warn!(error = %error, "dashboard server closed with an error");
            }
            Ok(Err(error)) => {
                tracing::warn!(error = %error, "dashboard server task ended with an error");
            }
            Err(_) => {
                tracing::warn!("dashboard server did not stop within 10 seconds; aborting task");
                server.abort();
            }
        }
    }

    app.camera().stop(&app).await;
    if let Some(inference) = inference {
        inference.shutdown().await;
    }
    cleanup_task.abort();
    if let Err(error) = app.database().shutdown().await {
        tracing::warn!(error = %error, "failed to shut the database down cleanly");
    }
    tracing::info!("shutdown complete");

    if let Some(error) = server_error {
        Err(error)
    } else {
        Ok(())
    }
}

fn spawn_screenshot_cleanup(app: AppState) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(3_600));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let shutdown = app.shutdown_token();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                _ = interval.tick() => {
                    let config = app.config();
                    let directory = config.screenshot_dir();
                    let retention_days = config.storage.retention_days;
                    let result = tokio::task::spawn_blocking(move || {
                        cleanup_screenshots(&directory, retention_days)
                    })
                    .await;
                    match result {
                        Ok(Ok(0)) => {}
                        Ok(Ok(removed)) => tracing::info!(
                            removed,
                            retention_days = config.storage.retention_days,
                            "removed expired detection screenshots"
                        ),
                        Ok(Err(error)) => tracing::warn!(error = %error, "screenshot cleanup failed"),
                        Err(error) => tracing::warn!(error = %error, "screenshot cleanup task failed"),
                    }
                }
            }
        }
    })
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = match signal(SignalKind::terminate()) {
            Ok(signal) => signal,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn parse_args() -> Result<PathBuf, String> {
    let mut config_path = std::env::var("YOLOCAM_CONFIG")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config.toml"));
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--config" | "-c" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--config requires a path".to_string())?;
                config_path = PathBuf::from(value);
            }
            "--help" | "-h" => {
                println!(
                    "yolocam {}\n\nUsage: yolocam [--config PATH]\n\nEnvironment:\n  YOLOCAM_CONFIG              Config file path (default: config.toml)\n  YOLOCAM_RTSP_URL            Overrides camera.rtsp_url\n  YOLOCAM_RTSP_USERNAME       Overrides camera.username\n  YOLOCAM_RTSP_PASSWORD       Overrides camera.password\n  YOLOCAM_MODEL_PATH          Overrides yolo.model_path (auto = hardware selection)\n  YOLOCAM_ONNXRUNTIME_PATH    Overrides yolo.onnxruntime_path\n  ORT_DYLIB_PATH              ONNX Runtime library path fallback\n  RUST_LOG                    tracing filter (default: yolocam=info,ort=warn)",
                    env!("CARGO_PKG_VERSION")
                );
                std::process::exit(0);
            }
            "--version" | "-V" => {
                println!("yolocam {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other} (try --help)")),
        }
    }
    Ok(config_path)
}
