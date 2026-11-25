use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::{SecondsFormat, Utc};
use tokio::sync::mpsc;

use crate::config::{Config, InferenceDevice, AUTO_MODEL_PATH};
use crate::database::NewDetection;
use crate::detection::{save_screenshot, ClassFilter, Detection};
use crate::frame::{next_analysis_deadline, VideoFrame};
use crate::hardware::{self, AutoReport};
use crate::models;
use crate::state::{AiState, AiStatus, AnalysisView, AppState, CameraState, InferenceCommand};
use crate::yolo::{decode_jpeg, YoloEngine};

pub struct InferenceWorker {
    handle: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl InferenceWorker {
    pub fn spawn(app: AppState) -> std::io::Result<Self> {
        let stop = app.inference_stop_flag();
        let worker_stop = Arc::clone(&stop);
        let (sender, receiver) = mpsc::channel(4);
        app.set_inference_control(sender);
        let handle = std::thread::Builder::new()
            .name("yolo-inference".to_string())
            .spawn(move || run_worker(app, worker_stop, receiver))?;
        Ok(Self {
            handle: Some(handle),
            stop,
        })
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    pub async fn shutdown(mut self) {
        self.request_stop();
        let Some(handle) = self.handle.take() else {
            return;
        };
        match tokio::time::timeout(
            Duration::from_secs(20),
            tokio::task::spawn_blocking(move || handle.join()),
        )
        .await
        {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(_))) => tracing::error!("inference thread panicked"),
            Ok(Err(error)) => tracing::error!(error = %error, "failed to join inference thread"),
            Err(_) => tracing::warn!("inference thread did not stop within 20 seconds"),
        }
    }
}

impl Drop for InferenceWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

struct LoadedModel {
    engine: YoloEngine,
    model_path: String,
    device: InferenceDevice,
    input_width: u32,
    input_height: u32,
    auto: Option<AutoReport>,
}

fn run_worker(
    app: AppState,
    stop: Arc<AtomicBool>,
    mut commands: mpsc::Receiver<InferenceCommand>,
) {
    let mut loaded = match initialize_model(&app, &stop) {
        Some(loaded) => loaded,
        None => return,
    };

    let mut last_sequence: u64 = 0;
    let mut next_due = Instant::now();
    let mut consecutive_errors: u32 = 0;
    let mut last_screenshot_log: Option<Instant> = None;

    loop {
        if stop.load(Ordering::Acquire) || app.is_shutting_down() {
            break;
        }

        while let Ok(command) = commands.try_recv() {
            match command {
                InferenceCommand::Rebenchmark => {
                    rebenchmark(&app, &stop, &mut loaded);
                    next_due = Instant::now();
                }
            }
        }

        if app.camera_status().state == CameraState::Stopped {
            sleep_interruptible(&stop, Duration::from_millis(100));
            continue;
        }

        let config = app.config();
        let interval = Duration::from_millis(config.yolo.interval_ms.max(100));
        let now = Instant::now();
        if now < next_due {
            let wait = (next_due - now).min(Duration::from_millis(100));
            sleep_interruptible(&stop, wait);
            continue;
        }

        let Some(frame) = app.frame_store().latest() else {
            sleep_interruptible(&stop, Duration::from_millis(50));
            continue;
        };
        if frame.sequence <= last_sequence {
            next_due = now + interval;
            continue;
        }
        last_sequence = frame.sequence;

        match analyze_frame(
            &app,
            &mut loaded.engine,
            &config,
            &frame,
            &mut last_screenshot_log,
        ) {
            Ok(Some(analysis)) => {
                consecutive_errors = 0;
                app.record_analysis(analysis);
                if app.ai_status().message.is_some() {
                    refresh_ready_status(&app, &loaded, None);
                }
            }
            Ok(None) => {
                consecutive_errors = 0;
            }
            Err(error) => {
                consecutive_errors = consecutive_errors.saturating_add(1);
                if consecutive_errors == 1 || consecutive_errors.is_multiple_of(10) {
                    tracing::warn!(error = %error, consecutive_errors, "analysis failed");
                }
                refresh_ready_status(&app, &loaded, Some(format!("inference failed: {error}")));
            }
        }

        next_due = next_analysis_deadline(next_due, interval, Instant::now());
    }

    tracing::info!("inference worker stopped");
}

fn initialize_model(app: &AppState, stop: &AtomicBool) -> Option<LoadedModel> {
    let config = app.config();
    let auto_enabled = config.yolo.model_path == AUTO_MODEL_PATH;

    if !auto_enabled {
        tracing::info!(
            model = %config.yolo.model_path,
            device = config.yolo.device.as_str(),
            "loading YOLO model"
        );
        return match load_engine(app, &config, PathBuf::from(&config.yolo.model_path), None) {
            Ok(loaded) => Some(loaded),
            Err(message) => {
                tracing::error!(error = %message, "unable to load YOLO model");
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
    }

    let device = config.yolo.device;
    let mut progress = |message: String| {
        app.set_ai_status(AiStatus {
            state: AiState::Benchmarking,
            message: Some(message),
            model_path: AUTO_MODEL_PATH.to_string(),
            device,
            input_width: None,
            input_height: None,
            auto: None,
        });
    };

    let report = match hardware::resolve(&config, false, stop, &mut progress) {
        Ok(report) => report,
        Err(error) => {
            let message = format!("automatic model selection failed: {error}");
            tracing::error!(error = %message);
            if let Some(legacy) = models::legacy_model() {
                tracing::warn!(
                    model = %legacy.display(),
                    "falling back to the local yolo11n model"
                );
                let warning = format!("{message}; using {}", legacy.display());
                return match load_engine(app, &config, legacy, None) {
                    Ok(mut loaded) => {
                        loaded.auto = None;
                        app.set_ai_status(AiStatus {
                            state: AiState::Ready,
                            message: Some(warning),
                            model_path: loaded.model_path.clone(),
                            device,
                            input_width: Some(loaded.input_width),
                            input_height: Some(loaded.input_height),
                            auto: None,
                        });
                        Some(loaded)
                    }
                    Err(error) => {
                        set_error_status(app, &message, error, &config);
                        None
                    }
                };
            }
            set_error_status(app, &message, String::new(), &config);
            return None;
        }
    };

    let path = PathBuf::from(&report.selected_model_path);
    match load_engine(app, &config, path.clone(), Some(report)) {
        Ok(loaded) => Some(loaded),
        Err(message) => {
            if let Some(legacy) = models::legacy_model() {
                if legacy != path {
                    tracing::warn!(
                        error = %message,
                        model = %legacy.display(),
                        "the automatically selected model failed; trying the local yolo11n fallback"
                    );
                    if let Ok(mut loaded) = load_engine(app, &config, legacy.clone(), None) {
                        loaded.auto = None;
                        app.set_ai_status(AiStatus {
                            state: AiState::Ready,
                            message: Some(format!(
                                "automatically selected model failed ({message}); using {}",
                                legacy.display()
                            )),
                            model_path: loaded.model_path.clone(),
                            device,
                            input_width: Some(loaded.input_width),
                            input_height: Some(loaded.input_height),
                            auto: None,
                        });
                        return Some(loaded);
                    }
                }
            }
            set_error_status(
                app,
                "unable to load the automatically selected model",
                message,
                &config,
            );
            None
        }
    }
}

fn rebenchmark(app: &AppState, stop: &AtomicBool, loaded: &mut LoadedModel) {
    let config = app.config();
    if config.yolo.model_path != AUTO_MODEL_PATH {
        refresh_ready_status(
            app,
            loaded,
            Some("Automatic model selection is disabled; set model_path = \"auto\".".to_string()),
        );
        return;
    }

    let device = config.yolo.device;
    let mut progress = |message: String| {
        app.set_ai_status(AiStatus {
            state: AiState::Benchmarking,
            message: Some(message),
            model_path: AUTO_MODEL_PATH.to_string(),
            device,
            input_width: None,
            input_height: None,
            auto: None,
        });
    };

    match hardware::resolve(&config, true, stop, &mut progress) {
        Ok(report) => {
            let path = PathBuf::from(&report.selected_model_path);
            match load_engine(app, &config, path, Some(report)) {
                Ok(new_loaded) => {
                    tracing::info!(
                        model = %new_loaded.model_path,
                        "automatic model selection updated"
                    );
                    *loaded = new_loaded;
                }
                Err(error) => {
                    tracing::error!(error = %error, "the re-selected model failed to load");
                    refresh_ready_status(
                        app,
                        loaded,
                        Some(format!(
                            "hardware test finished, but the selected model failed: {error}"
                        )),
                    );
                }
            }
        }
        Err(error) => {
            tracing::warn!(error = %error, "hardware re-test failed");
            refresh_ready_status(app, loaded, Some(format!("hardware test failed: {error}")));
        }
    }
}

fn load_engine(
    app: &AppState,
    config: &Config,
    path: PathBuf,
    auto: Option<AutoReport>,
) -> Result<LoadedModel, String> {
    let mut yolo = config.yolo.clone();
    yolo.model_path = path.display().to_string();
    tracing::info!(
        model = %yolo.model_path,
        device = yolo.device.as_str(),
        "loading YOLO model"
    );
    let engine = YoloEngine::load(&yolo).map_err(|error| error.to_string())?;
    let (input_width, input_height) = engine.input_size();
    let loaded = LoadedModel {
        engine,
        model_path: yolo.model_path.clone(),
        device: yolo.device,
        input_width,
        input_height,
        auto,
    };
    tracing::info!(
        input_width,
        input_height,
        device = loaded.device.as_str(),
        model = %loaded.model_path,
        "YOLO model ready"
    );
    app.set_ai_status(AiStatus {
        state: AiState::Ready,
        message: None,
        model_path: loaded.model_path.clone(),
        device: loaded.device,
        input_width: Some(input_width),
        input_height: Some(input_height),
        auto: loaded.auto.clone(),
    });
    Ok(loaded)
}

fn set_error_status(app: &AppState, message: &str, detail: String, config: &Config) {
    let message = if detail.is_empty() {
        message.to_string()
    } else {
        format!("{message}: {detail}")
    };
    tracing::error!(error = %message, "unable to load the selected YOLO model");
    app.set_ai_status(AiStatus {
        state: AiState::Error,
        message: Some(message),
        model_path: config.yolo.model_path.clone(),
        device: config.yolo.device,
        input_width: None,
        input_height: None,
        auto: None,
    });
}

fn analyze_frame(
    app: &AppState,
    engine: &mut YoloEngine,
    config: &crate::config::Config,
    frame: &VideoFrame,
    last_screenshot_log: &mut Option<Instant>,
) -> Result<Option<AnalysisView>, String> {
    let pipeline_start = Instant::now();
    let image = decode_jpeg(&frame.jpeg).map_err(|error| error.to_string())?;
    let filter = ClassFilter {
        confidence_threshold: config.yolo.confidence_threshold,
        detect_person: config.yolo.detect_person,
        detect_car: config.yolo.detect_car,
    };
    let inference_start = Instant::now();
    let detections = engine
        .infer(&image, filter)
        .map_err(|error| error.to_string())?;
    let inference_ms = inference_start.elapsed().as_secs_f32() * 1_000.0;
    let pipeline_ms = pipeline_start.elapsed().as_secs_f32() * 1_000.0;

    // A stop or reconnect can happen while decoding/inference runs. Do not
    // publish stale boxes for a camera that is no longer streaming.
    if app.camera_status().state != CameraState::Streaming {
        return Ok(None);
    }

    let analyzed_at = Utc::now();
    let timestamp = analyzed_at.to_rfc3339_opts(SecondsFormat::Millis, true);
    let screenshot = save_frame_screenshot(
        config,
        &image,
        &detections,
        analyzed_at,
        last_screenshot_log,
    );
    enqueue_detection_events(
        app,
        config,
        &detections,
        &timestamp,
        analyzed_at.timestamp_millis(),
        frame.sequence,
        screenshot.as_deref(),
    );

    let (people, cars) = crate::detection::count_by_class(&detections);
    Ok(Some(AnalysisView {
        analyzed_at: timestamp,
        frame_sequence: frame.sequence,
        inference_ms,
        pipeline_ms,
        source_width: image.width(),
        source_height: image.height(),
        people,
        cars,
        detections,
        screenshot,
    }))
}

fn save_frame_screenshot(
    config: &crate::config::Config,
    image: &image::RgbImage,
    detections: &[Detection],
    analyzed_at: chrono::DateTime<Utc>,
    last_log: &mut Option<Instant>,
) -> Option<String> {
    if !config.storage.save_screenshots || detections.is_empty() {
        return None;
    }
    let directory = config.screenshot_dir();
    match save_screenshot(
        image,
        detections,
        &config.camera.name,
        analyzed_at,
        &directory,
    ) {
        Ok(filename) => Some(filename),
        Err(error) => {
            let now = Instant::now();
            let should_log = last_log
                .map(|last| now.duration_since(last) >= Duration::from_secs(10))
                .unwrap_or(true);
            if should_log {
                tracing::warn!(
                    directory = %directory.display(),
                    error = %error,
                    "failed to save detection screenshot"
                );
                *last_log = Some(now);
            }
            None
        }
    }
}

fn enqueue_detection_events(
    app: &AppState,
    config: &crate::config::Config,
    detections: &[Detection],
    timestamp: &str,
    epoch_ms: i64,
    frame_sequence: u64,
    screenshot: Option<&str>,
) {
    for detection in detections {
        let record = NewDetection {
            timestamp: timestamp.to_string(),
            epoch_ms,
            class: detection.class,
            confidence: detection.confidence,
            camera_name: config.camera.name.clone(),
            screenshot: screenshot.map(str::to_string),
            frame_sequence,
        };
        if !app.database().try_enqueue(record) {
            break;
        }
    }
}

fn refresh_ready_status(app: &AppState, loaded: &LoadedModel, message: Option<String>) {
    let current = app.ai_status();
    if current.message == message
        && current.state == AiState::Ready
        && current.input_width == Some(loaded.input_width)
        && current.input_height == Some(loaded.input_height)
        && current.model_path == loaded.model_path
        && current.device == loaded.device
    {
        return;
    }
    app.set_ai_status(AiStatus {
        state: AiState::Ready,
        message,
        model_path: loaded.model_path.clone(),
        device: loaded.device,
        input_width: Some(loaded.input_width),
        input_height: Some(loaded.input_height),
        auto: loaded.auto.clone(),
    });
}

fn sleep_interruptible(stop: &AtomicBool, duration: Duration) {
    let deadline = Instant::now() + duration;
    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = (deadline - now).min(Duration::from_millis(50));
        std::thread::sleep(remaining);
    }
}
