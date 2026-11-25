use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use chrono::{SecondsFormat, Utc};
use serde::Serialize;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::{Config, InferenceDevice};
use crate::database::{Database, StoredCounts};
use crate::detection::{count_by_class, Detection};
use crate::frame::FrameStore;
use crate::rtsp::CameraManager;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraState {
    Stopped,
    Connecting,
    Streaming,
    Reconnecting,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CameraStatus {
    pub state: CameraState,
    pub message: Option<String>,
    pub attempt: u32,
    pub retry_in_seconds: Option<u64>,
}

impl Default for CameraStatus {
    fn default() -> Self {
        Self {
            state: CameraState::Stopped,
            message: None,
            attempt: 0,
            retry_in_seconds: None,
        }
    }
}

impl CameraStatus {
    pub fn stopped() -> Self {
        Self::default()
    }

    pub fn connecting() -> Self {
        Self {
            state: CameraState::Connecting,
            message: Some("Connecting to RTSP stream".to_string()),
            attempt: 0,
            retry_in_seconds: None,
        }
    }

    pub fn streaming() -> Self {
        Self {
            state: CameraState::Streaming,
            message: None,
            attempt: 0,
            retry_in_seconds: None,
        }
    }

    pub fn reconnecting(attempt: u32, retry_in_seconds: u64, message: String) -> Self {
        Self {
            state: CameraState::Reconnecting,
            message: Some(message),
            attempt,
            retry_in_seconds: Some(retry_in_seconds),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            state: CameraState::Error,
            message: Some(message),
            attempt: 0,
            retry_in_seconds: None,
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.state == CameraState::Streaming
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AiState {
    Loading,
    Benchmarking,
    Ready,
    Error,
    Stopped,
}

#[derive(Debug, Clone, Serialize)]
pub struct AiStatus {
    pub state: AiState,
    pub message: Option<String>,
    pub model_path: String,
    pub device: InferenceDevice,
    pub input_width: Option<u32>,
    pub input_height: Option<u32>,
    pub auto: Option<crate::hardware::AutoReport>,
}

impl AiStatus {
    pub fn loading(model_path: String, device: InferenceDevice) -> Self {
        Self {
            state: AiState::Loading,
            message: Some("Loading YOLO model".to_string()),
            model_path,
            device,
            input_width: None,
            input_height: None,
            auto: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceCommand {
    Rebenchmark,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisView {
    pub analyzed_at: String,
    pub frame_sequence: u64,
    pub inference_ms: f32,
    pub pipeline_ms: f32,
    pub source_width: u32,
    pub source_height: u32,
    pub people: usize,
    pub cars: usize,
    pub detections: Vec<Detection>,
    pub screenshot: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct TotalsView {
    pub people_events: u64,
    pub car_events: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraRuntime {
    pub name: String,
    pub status: CameraStatus,
    pub fps: f32,
    pub frame_sequence: u64,
    pub last_frame_at: Option<String>,
    pub frames_received: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveSnapshot {
    pub server_time: String,
    pub uptime_seconds: u64,
    pub camera: CameraRuntime,
    pub analysis: Option<AnalysisView>,
    pub totals: TotalsView,
    pub last_detection_at: Option<String>,
    pub ai: AiStatus,
    pub detection_interval_ms: u64,
    pub confidence_threshold: f32,
    pub enabled_classes: EnabledClasses,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnabledClasses {
    pub person: bool,
    pub car: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum WsEvent {
    Snapshot(Box<LiveSnapshot>),
}

#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppInner>,
}

struct AppInner {
    config: RwLock<Config>,
    config_path: PathBuf,
    frame_store: FrameStore,
    camera_status: RwLock<CameraStatus>,
    analysis: RwLock<Option<AnalysisView>>,
    ai: RwLock<AiStatus>,
    last_detection_at: RwLock<Option<String>>,
    people_events: AtomicU64,
    car_events: AtomicU64,
    database: Database,
    camera_manager: CameraManager,
    events: broadcast::Sender<WsEvent>,
    shutdown: CancellationToken,
    inference_stop: Arc<AtomicBool>,
    inference_control: RwLock<Option<mpsc::Sender<InferenceCommand>>>,
    started_at: Instant,
}

impl AppState {
    pub fn new(
        config: Config,
        config_path: PathBuf,
        database: Database,
        stored_counts: StoredCounts,
        last_detection_at: Option<String>,
    ) -> Self {
        let ai = AiStatus::loading(config.yolo.model_path.clone(), config.yolo.device);
        let (events, _) = broadcast::channel(64);
        let inner = Arc::new(AppInner {
            config: RwLock::new(config),
            config_path,
            frame_store: FrameStore::new(),
            camera_status: RwLock::new(CameraStatus::stopped()),
            analysis: RwLock::new(None),
            ai: RwLock::new(ai),
            last_detection_at: RwLock::new(last_detection_at),
            people_events: AtomicU64::new(stored_counts.people),
            car_events: AtomicU64::new(stored_counts.cars),
            database,
            camera_manager: CameraManager::new(),
            events,
            shutdown: CancellationToken::new(),
            inference_stop: Arc::new(AtomicBool::new(false)),
            inference_control: RwLock::new(None),
            started_at: Instant::now(),
        });
        Self { inner }
    }

    pub fn config(&self) -> Config {
        self.inner
            .config
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn replace_config(&self, config: Config) {
        *self
            .inner
            .config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = config;
    }

    pub fn config_path(&self) -> &PathBuf {
        &self.inner.config_path
    }

    pub fn frame_store(&self) -> &FrameStore {
        &self.inner.frame_store
    }

    pub fn database(&self) -> &Database {
        &self.inner.database
    }

    pub fn camera(&self) -> &CameraManager {
        &self.inner.camera_manager
    }

    pub fn shutdown_token(&self) -> CancellationToken {
        self.inner.shutdown.clone()
    }

    pub fn inference_stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.inner.inference_stop)
    }

    pub fn set_inference_control(&self, sender: mpsc::Sender<InferenceCommand>) {
        *self
            .inner
            .inference_control
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sender);
    }

    /// Ask the inference worker to run the hardware test and load the
    /// resulting model without restarting the process.
    pub fn request_rebenchmark(&self) -> bool {
        let sender = self
            .inner
            .inference_control
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        match sender {
            Some(sender) => sender.try_send(InferenceCommand::Rebenchmark).is_ok(),
            None => false,
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        self.inner.shutdown.is_cancelled()
    }

    pub fn request_shutdown(&self) {
        self.inner.inference_stop.store(true, Ordering::Release);
        self.inner.shutdown.cancel();
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<WsEvent> {
        self.inner.events.subscribe()
    }

    pub fn camera_status(&self) -> CameraStatus {
        self.inner
            .camera_status
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn set_camera_status(&self, status: CameraStatus) {
        let changed = {
            let mut current = self
                .inner
                .camera_status
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *current == status {
                false
            } else {
                *current = status;
                true
            }
        };
        if changed {
            self.publish_snapshot();
        }
    }

    pub fn ai_status(&self) -> AiStatus {
        self.inner
            .ai
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn set_ai_status(&self, status: AiStatus) {
        *self
            .inner
            .ai
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = status;
        self.publish_snapshot();
    }

    pub fn analysis(&self) -> Option<AnalysisView> {
        self.inner
            .analysis
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn clear_analysis(&self) {
        *self
            .inner
            .analysis
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// Store the latest analyzed frame, update cumulative event totals and
    /// last-detection time, then notify WebSocket clients.
    pub fn record_analysis(&self, analysis: AnalysisView) {
        let (people, cars) = count_by_class(&analysis.detections);
        if people > 0 {
            self.inner
                .people_events
                .fetch_add(people as u64, Ordering::Relaxed);
        }
        if cars > 0 {
            self.inner
                .car_events
                .fetch_add(cars as u64, Ordering::Relaxed);
        }
        if !analysis.detections.is_empty() {
            *self
                .inner
                .last_detection_at
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(analysis.analyzed_at.clone());
        }
        *self
            .inner
            .analysis
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(analysis);
        self.publish_snapshot();
    }

    pub fn publish_snapshot(&self) {
        let snapshot = self.snapshot();
        let _ = self
            .inner
            .events
            .send(WsEvent::Snapshot(Box::new(snapshot)));
    }

    pub fn snapshot(&self) -> LiveSnapshot {
        let config = self.config();
        let status = self.camera_status();
        let analysis = self.analysis();
        let ai = self.ai_status();
        let last_detection_at = self
            .inner
            .last_detection_at
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let frame_stats = self.inner.frame_store.stats();
        let totals = TotalsView {
            people_events: self.inner.people_events.load(Ordering::Relaxed),
            car_events: self.inner.car_events.load(Ordering::Relaxed),
        };

        LiveSnapshot {
            server_time: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
            uptime_seconds: self.inner.started_at.elapsed().as_secs(),
            camera: CameraRuntime {
                name: config.camera.name.clone(),
                status,
                fps: frame_stats.camera_fps,
                frame_sequence: frame_stats.sequence,
                last_frame_at: frame_stats.last_frame_at,
                frames_received: frame_stats.frames_received,
            },
            analysis,
            totals,
            last_detection_at,
            ai,
            detection_interval_ms: config.yolo.interval_ms,
            confidence_threshold: config.yolo.confidence_threshold,
            enabled_classes: EnabledClasses {
                person: config.yolo.detect_person,
                car: config.yolo.detect_car,
            },
        }
    }
}
