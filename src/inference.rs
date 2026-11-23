use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chrono::{SecondsFormat, Utc};

use crate::config::InferenceDevice;
use crate::database::NewDetection;
use crate::detection::{save_screenshot, ClassFilter, Detection};
use crate::frame::{next_analysis_deadline, VideoFrame};
use crate::state::{AiState, AiStatus, AnalysisView, AppState, CameraState};
use crate::yolo::{decode_jpeg, YoloEngine};

pub struct InferenceWorker {
    handle: Option<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl InferenceWorker {
    pub fn spawn(app: AppState) -> std::io::Result<Self> {
        let stop = app.inference_stop_flag();
        let worker_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("yolo-inference".to_string())
            .spawn(move || run_worker(app, worker_stop))?;
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

fn run_worker(app: AppState, stop: Arc<AtomicBool>) {
    let config = app.config();
    tracing::info!(
        model = %config.yolo.model_path,
        device = config.yolo.device.as_str(),
        "loading YOLO model"
    );
    let mut engine = match YoloEngine::load(&config.yolo) {
        Ok(engine) => engine,
        Err(error) => {
            let message = error.to_string();
            tracing::error!(error = %message, "unable to load YOLO model");
            app.set_ai_status(AiStatus {
                state: AiState::Error,
                message: Some(message),
                model_path: config.yolo.model_path.clone(),
                device: config.yolo.device,
                input_width: None,
                input_height: None,
            });
            return;
        }
    };

    let (input_width, input_height) = engine.input_size();
    let loaded_model_path = config.yolo.model_path.clone();
    let loaded_device = config.yolo.device;
    tracing::info!(
        input_width,
        input_height,
        device = loaded_device.as_str(),
        "YOLO model ready"
    );
    app.set_ai_status(AiStatus {
        state: AiState::Ready,
        message: None,
        model_path: loaded_model_path.clone(),
        device: loaded_device,
        input_width: Some(input_width),
        input_height: Some(input_height),
    });

    let mut last_sequence: u64 = 0;
    let mut next_due = Instant::now();
    let mut consecutive_errors: u32 = 0;
    let mut last_screenshot_log: Option<Instant> = None;

    loop {
        if stop.load(Ordering::Acquire) || app.is_shutting_down() {
            break;
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

        match analyze_frame(&app, &mut engine, &config, &frame, &mut last_screenshot_log) {
            Ok(Some(analysis)) => {
                consecutive_errors = 0;
                app.record_analysis(analysis);
                if app
                    .ai_status()
                    .message
                    .as_deref()
                    .is_some_and(|message| message.starts_with("inference failed"))
                {
                    refresh_ready_status(
                        &app,
                        &loaded_model_path,
                        loaded_device,
                        input_width,
                        input_height,
                        None,
                    );
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
                refresh_ready_status(
                    &app,
                    &loaded_model_path,
                    loaded_device,
                    input_width,
                    input_height,
                    Some(format!("inference failed: {error}")),
                );
            }
        }

        next_due = next_analysis_deadline(next_due, interval, Instant::now());
    }

    tracing::info!("inference worker stopped");
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

fn refresh_ready_status(
    app: &AppState,
    loaded_model_path: &str,
    loaded_device: InferenceDevice,
    input_width: u32,
    input_height: u32,
    message: Option<String>,
) {
    let current = app.ai_status();
    if current.message == message
        && current.state == AiState::Ready
        && current.input_width == Some(input_width)
        && current.input_height == Some(input_height)
        && current.model_path == loaded_model_path
        && current.device == loaded_device
    {
        return;
    }
    app.set_ai_status(AiStatus {
        state: AiState::Ready,
        message,
        model_path: loaded_model_path.to_string(),
        device: loaded_device,
        input_width: Some(input_width),
        input_height: Some(input_height),
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
