use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

const DEFAULT_CONFIDENCE: f32 = 0.5;
const DEFAULT_INTERVAL_MS: u64 = 1_000;
const MIN_INTERVAL_MS: u64 = 100;
const MAX_INTERVAL_MS: u64 = 60_000;
const DEFAULT_PREVIEW_MAX_DIMENSION: u32 = 1_280;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read configuration: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid configuration file: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera: CameraConfig,
    pub yolo: YoloConfig,
    pub storage: StorageConfig,
    pub server: ServerConfig,
    /// True when RTSP URL or credentials came from environment variables.
    /// Such values are never written back to disk by the settings API.
    #[serde(skip)]
    pub env_camera_locked: bool,
    /// True only when the credentials themselves came from the environment;
    /// controls whether username/password are stripped when saving.
    #[serde(skip)]
    pub env_credentials_locked: bool,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("camera", &self.camera)
            .field("yolo", &self.yolo)
            .field("storage", &self.storage)
            .field("server", &self.server)
            .field("env_camera_locked", &self.env_camera_locked)
            .field("env_credentials_locked", &self.env_credentials_locked)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    pub name: String,
    /// RTSP URL without userinfo. Credentials are kept separately so they can
    /// be redacted reliably and supplied through environment variables.
    pub rtsp_url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    pub reconnect_delay_seconds: u64,
    pub ffmpeg_path: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            name: "Camera".to_string(),
            rtsp_url: String::new(),
            username: None,
            password: None,
            reconnect_delay_seconds: 5,
            ffmpeg_path: "ffmpeg".to_string(),
        }
    }
}

impl fmt::Debug for CameraConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CameraConfig")
            .field("name", &self.name)
            .field("rtsp_url", &redact_url_string(&self.rtsp_url))
            .field(
                "username",
                &if self.username.is_some() {
                    "<redacted>"
                } else {
                    "<none>"
                },
            )
            .field(
                "password",
                &if self.password.is_some() {
                    "<redacted>"
                } else {
                    "<none>"
                },
            )
            .field("reconnect_delay_seconds", &self.reconnect_delay_seconds)
            .field("ffmpeg_path", &self.ffmpeg_path)
            .finish()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct YoloConfig {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub interval_ms: u64,
    pub detect_person: bool,
    pub detect_car: bool,
    pub device: InferenceDevice,
    /// Optional explicit path to onnxruntime.dll / libonnxruntime.so.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnxruntime_path: Option<String>,
}

impl Default for YoloConfig {
    fn default() -> Self {
        Self {
            model_path: "models/yolo11n.onnx".to_string(),
            confidence_threshold: DEFAULT_CONFIDENCE,
            interval_ms: DEFAULT_INTERVAL_MS,
            detect_person: true,
            detect_car: true,
            device: InferenceDevice::Cpu,
            onnxruntime_path: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceDevice {
    #[default]
    Cpu,
    Cuda,
    #[serde(rename = "directml")]
    DirectMl,
}

impl InferenceDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::DirectMl => "directml",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub save_screenshots: bool,
    pub screenshot_directory: String,
    pub retention_days: u32,
    pub database_path: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            save_screenshots: true,
            screenshot_directory: "./detections".to_string(),
            retention_days: 7,
            database_path: "./yolocam.sqlite3".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

impl ServerConfig {
    pub fn bind_address(&self) -> String {
        // IPv6 literals need brackets when combined with a port.
        if self.host.contains(':') && !self.host.starts_with('[') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

#[derive(Default, Clone)]
pub struct EnvOverrides {
    pub rtsp_url: Option<String>,
    pub username: Option<String>,
    pub password: Option<String>,
    pub model_path: Option<String>,
    pub onnxruntime_path: Option<String>,
}

impl EnvOverrides {
    pub fn from_process() -> Self {
        fn read(name: &str) -> Option<String> {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        }

        Self {
            rtsp_url: read("YOLOCAM_RTSP_URL"),
            username: read("YOLOCAM_RTSP_USERNAME"),
            password: read("YOLOCAM_RTSP_PASSWORD"),
            model_path: read("YOLOCAM_MODEL_PATH"),
            onnxruntime_path: read("YOLOCAM_ONNXRUNTIME_PATH"),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = fs::read_to_string(path)?;
        let mut config: Self = toml::from_str(&text)?;
        config.normalize_credentials()?;
        config.apply_env_overrides(EnvOverrides::from_process())?;
        config.validate()?;
        Ok(config)
    }

    #[cfg(test)]
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let mut config: Self = toml::from_str(text)?;
        config.normalize_credentials()?;
        config.validate()?;
        Ok(config)
    }

    pub fn apply_env_overrides(&mut self, env: EnvOverrides) -> Result<(), ConfigError> {
        let mut camera_locked = false;
        let mut credentials_locked = false;

        if let Some(raw_url) = env.rtsp_url {
            let mut url = Url::parse(&raw_url)
                .map_err(|_| ConfigError::Invalid("YOLOCAM_RTSP_URL is not a valid URL".into()))?;
            if !matches!(url.scheme(), "rtsp" | "rtsps") {
                return Err(ConfigError::Invalid(
                    "YOLOCAM_RTSP_URL must use rtsp:// or rtsps://".into(),
                ));
            }
            if let Some((username, password)) = take_url_credentials(&mut url)? {
                self.camera.username = Some(username);
                self.camera.password = password;
                self.env_camera_locked = true;
                credentials_locked = true;
            }
            self.camera.rtsp_url = url.to_string();
            camera_locked = true;
        }

        if let Some(username) = env.username {
            self.camera.username = Some(username);
            self.env_camera_locked = true;
            credentials_locked = true;
            camera_locked = true;
        }
        if let Some(password) = env.password {
            self.camera.password = Some(password);
            self.env_camera_locked = true;
            credentials_locked = true;
            camera_locked = true;
        }
        if let Some(model_path) = env.model_path {
            self.yolo.model_path = model_path;
        }
        if let Some(runtime_path) = env.onnxruntime_path {
            self.yolo.onnxruntime_path = Some(runtime_path);
        }

        self.env_camera_locked |= camera_locked;
        self.env_credentials_locked |= credentials_locked;
        Ok(())
    }

    /// Move userinfo embedded in the RTSP URL into separate credential fields
    /// and remove it from the URL. This keeps it out of logs and API output.
    pub fn normalize_credentials(&mut self) -> Result<(), ConfigError> {
        let mut url = Url::parse(&self.camera.rtsp_url).map_err(|_| {
            ConfigError::Invalid(
                "camera.rtsp_url is not a valid URL (example: rtsp://192.168.1.100:554/stream1)"
                    .into(),
            )
        })?;
        // Scheme validation happens in Config::validate so every problem can
        // be reported in one message. Normalization only removes userinfo.
        if let Some((username, password)) = take_url_credentials(&mut url)? {
            self.camera.username = Some(username);
            self.camera.password = password;
        }
        self.camera.rtsp_url = url.to_string();
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut problems: Vec<String> = Vec::new();

        if self.camera.name.trim().is_empty() {
            problems.push("camera.name must not be empty".into());
        } else if self.camera.name.chars().count() > 64 {
            problems.push("camera.name must be at most 64 characters".into());
        }

        match Url::parse(&self.camera.rtsp_url) {
            Ok(url) => {
                if !matches!(url.scheme(), "rtsp" | "rtsps") {
                    problems.push("camera.rtsp_url must use rtsp:// or rtsps://".into());
                }
                if url.host_str().is_none() {
                    problems.push("camera.rtsp_url must contain a host".into());
                }
                if !url.username().is_empty() || url.password().is_some() {
                    problems.push(
                        "camera.rtsp_url must not contain credentials; use YOLOCAM_RTSP_USERNAME / YOLOCAM_RTSP_PASSWORD or the separate camera.username and camera.password fields"
                            .into(),
                    );
                }
            }
            Err(_) => problems.push(
                "camera.rtsp_url is not a valid URL (example: rtsp://192.168.1.100:554/stream1)"
                    .into(),
            ),
        }

        if self.camera.username.as_deref() == Some("") && self.camera.password.is_some() {
            problems.push("camera.password requires a non-empty camera.username".into());
        }
        if !(1..=300).contains(&self.camera.reconnect_delay_seconds) {
            problems.push("camera.reconnect_delay_seconds must be between 1 and 300".into());
        }
        if self.camera.ffmpeg_path.trim().is_empty() {
            problems.push("camera.ffmpeg_path must not be empty".into());
        }

        if self.yolo.model_path.trim().is_empty() {
            problems.push("yolo.model_path must not be empty".into());
        }
        if !self.yolo.confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.yolo.confidence_threshold)
        {
            problems.push("yolo.confidence_threshold must be a number between 0.0 and 1.0".into());
        }
        if !(MIN_INTERVAL_MS..=MAX_INTERVAL_MS).contains(&self.yolo.interval_ms) {
            problems.push(format!(
                "yolo.interval_ms must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS}"
            ));
        }
        if !self.yolo.detect_person && !self.yolo.detect_car {
            problems.push(
                "at least one of yolo.detect_person or yolo.detect_car must be enabled".into(),
            );
        }
        match self.yolo.device {
            InferenceDevice::Cpu => {}
            InferenceDevice::Cuda if !cfg!(feature = "cuda") => {
                problems.push("yolo.device = \"cuda\" requires a build with --features cuda".into())
            }
            InferenceDevice::DirectMl if !cfg!(feature = "directml") => problems.push(
                "yolo.device = \"directml\" requires a build with --features directml".into(),
            ),
            InferenceDevice::DirectMl if !cfg!(windows) => {
                problems.push("yolo.device = \"directml\" is only available on Windows".into())
            }
            _ => {}
        }

        if self.storage.screenshot_directory.trim().is_empty() {
            problems.push("storage.screenshot_directory must not be empty".into());
        }
        if !(1..=3_650).contains(&self.storage.retention_days) {
            problems.push("storage.retention_days must be between 1 and 3650".into());
        }
        if self.storage.database_path.trim().is_empty() {
            problems.push("storage.database_path must not be empty".into());
        }

        if self.server.host.trim().is_empty() {
            problems.push("server.host must not be empty".into());
        }
        if self.server.port == 0 {
            problems.push("server.port must be greater than zero".into());
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(problems.join("; ")))
        }
    }

    /// Build the URL passed to FFmpeg, including credentials. Never log this.
    pub fn effective_rtsp_url(&self) -> Result<Url, ConfigError> {
        let mut url = Url::parse(&self.camera.rtsp_url)
            .map_err(|_| ConfigError::Invalid("camera.rtsp_url is not a valid URL".into()))?;
        if !matches!(url.scheme(), "rtsp" | "rtsps") {
            return Err(ConfigError::Invalid(
                "camera.rtsp_url must use rtsp:// or rtsps://".into(),
            ));
        }
        if let Some(username) = &self.camera.username {
            url.set_username(username).map_err(|_| {
                ConfigError::Invalid("camera.username cannot be encoded in a URL".into())
            })?;
        }
        if let Some(password) = &self.camera.password {
            url.set_password(Some(password)).map_err(|_| {
                ConfigError::Invalid("camera.password cannot be encoded in a URL".into())
            })?;
        }
        Ok(url)
    }

    /// URL safe for logs and API responses (no username or password).
    pub fn public_rtsp_url(&self) -> Result<String, ConfigError> {
        let mut url = self.effective_rtsp_url()?;
        url.set_username("")
            .map_err(|_| ConfigError::Invalid("cannot strip RTSP username".into()))?;
        url.set_password(None)
            .map_err(|_| ConfigError::Invalid("cannot strip RTSP password".into()))?;
        Ok(url.to_string())
    }

    pub fn secrets(&self) -> Vec<String> {
        let mut secrets = Vec::new();
        if let Ok(url) = self.effective_rtsp_url() {
            if !url.username().is_empty() || url.password().is_some() {
                secrets.push(url.to_string());
            }
        }
        if let Some(username) = &self.camera.username {
            if !username.is_empty() {
                secrets.push(username.clone());
            }
        }
        if let Some(password) = &self.camera.password {
            if !password.is_empty() {
                secrets.push(password.clone());
            }
        }
        secrets
    }

    pub fn screenshot_dir(&self) -> PathBuf {
        PathBuf::from(&self.storage.screenshot_directory)
    }

    pub fn database_path(&self) -> PathBuf {
        PathBuf::from(&self.storage.database_path)
    }

    pub fn preview_max_dimension(&self) -> u32 {
        DEFAULT_PREVIEW_MAX_DIMENSION
    }

    pub fn apply_update(&mut self, update: &SettingsUpdate) -> Result<UpdateOutcome, ConfigError> {
        let mut outcome = UpdateOutcome::default();
        let mut restart = HashSet::new();

        if let Some(camera) = &update.camera {
            if self.env_camera_locked
                && (camera.rtsp_url.is_some()
                    || camera.username.is_some()
                    || camera.password.is_some()
                    || camera.clear_credentials)
            {
                return Err(ConfigError::Invalid(
                    "RTSP settings are managed by environment variables and cannot be changed here"
                        .into(),
                ));
            }

            if let Some(name) = &camera.name {
                let name = name.trim();
                if name.is_empty() {
                    return Err(ConfigError::Invalid("camera.name must not be empty".into()));
                }
                if name.chars().count() > 64 {
                    return Err(ConfigError::Invalid(
                        "camera.name must be at most 64 characters".into(),
                    ));
                }
                if self.camera.name != name {
                    self.camera.name = name.to_string();
                }
            }

            if let Some(raw_url) = &camera.rtsp_url {
                let raw_url = raw_url.trim();
                if raw_url.is_empty() {
                    return Err(ConfigError::Invalid(
                        "camera.rtsp_url must not be empty".into(),
                    ));
                }
                let mut new_url = Url::parse(raw_url).map_err(|_| {
                    ConfigError::Invalid(
                        "camera.rtsp_url is not a valid URL (example: rtsp://192.168.1.100:554/stream1)"
                            .into(),
                    )
                })?;
                if !matches!(new_url.scheme(), "rtsp" | "rtsps") {
                    return Err(ConfigError::Invalid(
                        "camera.rtsp_url must use rtsp:// or rtsps://".into(),
                    ));
                }
                let embedded = take_url_credentials(&mut new_url)?;
                let old_host = Url::parse(&self.camera.rtsp_url)
                    .ok()
                    .and_then(|url| url.host_str().map(str::to_string));
                let new_host = new_url.host_str().map(str::to_string);
                let host_changed = old_host != new_host;

                if self.camera.rtsp_url != new_url.as_str() {
                    self.camera.rtsp_url = new_url.to_string();
                    outcome.camera_connection_changed = true;
                }
                if let Some((username, password)) = embedded {
                    self.camera.username = Some(username);
                    self.camera.password = password;
                    outcome.camera_connection_changed = true;
                } else if host_changed {
                    // Never reuse credentials intended for a different camera host.
                    let had_username = self.camera.username.take().is_some();
                    let had_password = self.camera.password.take().is_some();
                    if had_username || had_password {
                        outcome.camera_connection_changed = true;
                    }
                }
            }

            if camera.clear_credentials {
                let had_username = self.camera.username.take().is_some();
                let had_password = self.camera.password.take().is_some();
                if had_username || had_password {
                    outcome.camera_connection_changed = true;
                }
            } else {
                if let Some(username) = &camera.username {
                    let username = username.trim();
                    if username.is_empty() {
                        let had_username = self.camera.username.take().is_some();
                        let had_password = self.camera.password.take().is_some();
                        outcome.camera_connection_changed |= had_username || had_password;
                    } else {
                        if self.camera.username.as_deref() != Some(username) {
                            self.camera.username = Some(username.to_string());
                            outcome.camera_connection_changed = true;
                        }
                    }
                }
                if let Some(password) = &camera.password {
                    if password.is_empty() {
                        if self.camera.password.take().is_some() {
                            outcome.camera_connection_changed = true;
                        }
                    } else if self.camera.password.as_deref() != Some(password) {
                        self.camera.password = Some(password.clone());
                        outcome.camera_connection_changed = true;
                    }
                }
                if self.camera.password.is_some() && self.camera.username.as_deref() == Some("") {
                    return Err(ConfigError::Invalid(
                        "camera.password requires a non-empty camera.username".into(),
                    ));
                }
            }

            if let Some(delay) = camera.reconnect_delay_seconds {
                if !(1..=300).contains(&delay) {
                    return Err(ConfigError::Invalid(
                        "camera.reconnect_delay_seconds must be between 1 and 300".into(),
                    ));
                }
                self.camera.reconnect_delay_seconds = delay;
            }

            if let Some(ffmpeg_path) = &camera.ffmpeg_path {
                let ffmpeg_path = ffmpeg_path.trim();
                if ffmpeg_path.is_empty() {
                    return Err(ConfigError::Invalid(
                        "camera.ffmpeg_path must not be empty".into(),
                    ));
                }
                if self.camera.ffmpeg_path != ffmpeg_path {
                    self.camera.ffmpeg_path = ffmpeg_path.to_string();
                    outcome.camera_connection_changed = true;
                }
            }
        }

        if let Some(yolo) = &update.yolo {
            if let Some(model_path) = &yolo.model_path {
                let model_path = model_path.trim();
                if model_path.is_empty() {
                    return Err(ConfigError::Invalid(
                        "yolo.model_path must not be empty".into(),
                    ));
                }
                if self.yolo.model_path != model_path {
                    self.yolo.model_path = model_path.to_string();
                    restart.insert("model_path");
                }
            }
            if let Some(threshold) = yolo.confidence_threshold {
                if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                    return Err(ConfigError::Invalid(
                        "yolo.confidence_threshold must be between 0.0 and 1.0".into(),
                    ));
                }
                self.yolo.confidence_threshold = threshold;
            }
            if let Some(interval) = yolo.interval_ms {
                if !(MIN_INTERVAL_MS..=MAX_INTERVAL_MS).contains(&interval) {
                    return Err(ConfigError::Invalid(format!(
                        "yolo.interval_ms must be between {MIN_INTERVAL_MS} and {MAX_INTERVAL_MS}"
                    )));
                }
                self.yolo.interval_ms = interval;
            }
            if let Some(enabled) = yolo.detect_person {
                self.yolo.detect_person = enabled;
            }
            if let Some(enabled) = yolo.detect_car {
                self.yolo.detect_car = enabled;
            }
            if !self.yolo.detect_person && !self.yolo.detect_car {
                return Err(ConfigError::Invalid(
                    "at least one of yolo.detect_person or yolo.detect_car must be enabled".into(),
                ));
            }
            if let Some(device) = yolo.device {
                match device {
                    InferenceDevice::Cpu => {}
                    InferenceDevice::Cuda if !cfg!(feature = "cuda") => {
                        return Err(ConfigError::Invalid(
                            "device \"cuda\" requires a build with --features cuda".into(),
                        ));
                    }
                    InferenceDevice::DirectMl if !cfg!(feature = "directml") => {
                        return Err(ConfigError::Invalid(
                            "device \"directml\" requires a build with --features directml".into(),
                        ));
                    }
                    InferenceDevice::DirectMl if !cfg!(windows) => {
                        return Err(ConfigError::Invalid(
                            "device \"directml\" is only available on Windows".into(),
                        ));
                    }
                    _ => {}
                }
                if self.yolo.device != device {
                    self.yolo.device = device;
                    restart.insert("device");
                }
            }
            if let Some(runtime_path) = &yolo.onnxruntime_path {
                let runtime_path = runtime_path.trim();
                let new_value = if runtime_path.is_empty() {
                    None
                } else {
                    Some(runtime_path.to_string())
                };
                if self.yolo.onnxruntime_path != new_value {
                    self.yolo.onnxruntime_path = new_value;
                    restart.insert("onnxruntime_path");
                }
            }
        }

        if let Some(storage) = &update.storage {
            if let Some(save) = storage.save_screenshots {
                self.storage.save_screenshots = save;
            }
            if let Some(directory) = &storage.screenshot_directory {
                let directory = directory.trim();
                if directory.is_empty() {
                    return Err(ConfigError::Invalid(
                        "storage.screenshot_directory must not be empty".into(),
                    ));
                }
                self.storage.screenshot_directory = directory.to_string();
            }
            if let Some(days) = storage.retention_days {
                if !(1..=3_650).contains(&days) {
                    return Err(ConfigError::Invalid(
                        "storage.retention_days must be between 1 and 3650".into(),
                    ));
                }
                self.storage.retention_days = days;
            }
        }

        let mut restart_required: Vec<&'static str> = restart.into_iter().collect();
        restart_required.sort_unstable();
        outcome.restart_required = restart_required.into_iter().map(str::to_string).collect();
        Ok(outcome)
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        let mut persisted = self.clone();
        if persisted.env_credentials_locked {
            // Environment-provided RTSP credentials are runtime-only.
            persisted.camera.username = None;
            persisted.camera.password = None;
        }
        persisted.env_camera_locked = false;
        persisted.env_credentials_locked = false;

        let text = toml::to_string_pretty(&persisted).map_err(|error| {
            ConfigError::Invalid(format!("cannot serialize configuration: {error}"))
        })?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let tmp = path.with_extension("toml.tmp");
        fs::write(&tmp, text)?;
        fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn public_settings(&self) -> PublicSettings {
        let rtsp_url = self
            .public_rtsp_url()
            .unwrap_or_else(|_| "<invalid URL>".to_string());
        PublicSettings {
            camera: PublicCameraSettings {
                name: self.camera.name.clone(),
                rtsp_url,
                username_set: self.camera.username.is_some(),
                password_set: self.camera.password.is_some(),
                env_locked: self.env_camera_locked,
                reconnect_delay_seconds: self.camera.reconnect_delay_seconds,
                ffmpeg_path: self.camera.ffmpeg_path.clone(),
            },
            yolo: PublicYoloSettings {
                model_path: self.yolo.model_path.clone(),
                confidence_threshold: self.yolo.confidence_threshold,
                interval_ms: self.yolo.interval_ms,
                detect_person: self.yolo.detect_person,
                detect_car: self.yolo.detect_car,
                device: self.yolo.device,
                onnxruntime_path: self.yolo.onnxruntime_path.clone(),
            },
            storage: self.storage.clone(),
            server: PublicServerSettings {
                host: self.server.host.clone(),
                port: self.server.port,
            },
            capabilities: Capabilities {
                cuda: cfg!(feature = "cuda"),
                directml: cfg!(feature = "directml"),
            },
        }
    }
}

fn take_url_credentials(url: &mut Url) -> Result<Option<(String, Option<String>)>, ConfigError> {
    let username = url.username().to_string();
    let password = url.password().map(str::to_string);
    if username.is_empty() && password.is_none() {
        return Ok(None);
    }
    let username = decode_url_component(&username)?;
    let password = match password {
        Some(password) => Some(decode_url_component(&password)?),
        None => None,
    };
    url.set_username("")
        .map_err(|_| ConfigError::Invalid("cannot remove RTSP username from URL".into()))?;
    url.set_password(None)
        .map_err(|_| ConfigError::Invalid("cannot remove RTSP password from URL".into()))?;
    Ok(Some((username, password)))
}

fn decode_url_component(value: &str) -> Result<String, ConfigError> {
    percent_decode_str(value)
        .decode_utf8()
        .map(|value| value.into_owned())
        .map_err(|_| ConfigError::Invalid("RTSP credentials are not valid UTF-8".into()))
}

fn redact_url_string(value: &str) -> String {
    let Ok(mut url) = Url::parse(value) else {
        return "<invalid URL>".to_string();
    };
    let had_credentials = !url.username().is_empty() || url.password().is_some();
    if had_credentials {
        let _ = url.set_username("");
        let _ = url.set_password(None);
        format!("{url} (credentials redacted)")
    } else {
        url.to_string()
    }
}

#[derive(Clone, Default)]
pub struct UpdateOutcome {
    pub camera_connection_changed: bool,
    pub restart_required: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct SettingsUpdate {
    pub camera: Option<CameraUpdate>,
    pub yolo: Option<YoloUpdate>,
    pub storage: Option<StorageUpdate>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct CameraUpdate {
    pub name: Option<String>,
    pub rtsp_url: Option<String>,
    /// `None` keeps the current value, an empty string clears it.
    pub username: Option<String>,
    /// `None` keeps the current value, an empty string clears it.
    pub password: Option<String>,
    #[serde(default)]
    pub clear_credentials: bool,
    pub reconnect_delay_seconds: Option<u64>,
    pub ffmpeg_path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct YoloUpdate {
    pub model_path: Option<String>,
    pub confidence_threshold: Option<f32>,
    pub interval_ms: Option<u64>,
    pub detect_person: Option<bool>,
    pub detect_car: Option<bool>,
    pub device: Option<InferenceDevice>,
    pub onnxruntime_path: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct StorageUpdate {
    pub save_screenshots: Option<bool>,
    pub screenshot_directory: Option<String>,
    pub retention_days: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicSettings {
    pub camera: PublicCameraSettings,
    pub yolo: PublicYoloSettings,
    pub storage: StorageConfig,
    pub server: PublicServerSettings,
    pub capabilities: Capabilities,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicCameraSettings {
    pub name: String,
    pub rtsp_url: String,
    pub username_set: bool,
    pub password_set: bool,
    pub env_locked: bool,
    pub reconnect_delay_seconds: u64,
    pub ffmpeg_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicYoloSettings {
    pub model_path: String,
    pub confidence_threshold: f32,
    pub interval_ms: u64,
    pub detect_person: bool,
    pub detect_car: bool,
    pub device: InferenceDevice,
    pub onnxruntime_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PublicServerSettings {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub cuda: bool,
    pub directml: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[camera]
name = "Front Camera"
rtsp_url = "rtsp://alice:s3cret@192.168.1.100:554/stream1"
reconnect_delay_seconds = 5

[yolo]
model_path = "models/yolo11n.onnx"
confidence_threshold = 0.50
interval_ms = 1000
detect_person = true
detect_car = true
device = "cpu"

[storage]
save_screenshots = true
screenshot_directory = "./detections"
retention_days = 7

[server]
host = "127.0.0.1"
port = 8080
"#;

    #[test]
    fn parses_example_config_and_extracts_credentials() {
        let config = Config::from_toml(SAMPLE).unwrap();
        assert_eq!(config.camera.name, "Front Camera");
        assert_eq!(config.camera.username.as_deref(), Some("alice"));
        assert_eq!(config.camera.password.as_deref(), Some("s3cret"));
        assert!(!config.camera.rtsp_url.contains("alice"));
        assert_eq!(
            config.public_rtsp_url().unwrap(),
            "rtsp://192.168.1.100:554/stream1"
        );
        assert_eq!(config.yolo.interval_ms, 1000);
        assert_eq!(config.storage.database_path, "./yolocam.sqlite3");
    }

    #[test]
    fn rejects_bad_values() {
        let text = SAMPLE
            .replace("confidence_threshold = 0.50", "confidence_threshold = 1.5")
            .replace("interval_ms = 1000", "interval_ms = 5")
            .replace("rtsp://alice:s3cret@", "ftp://");
        let error = Config::from_toml(&text).unwrap_err().to_string();
        assert!(error.contains("confidence_threshold"), "{error}");
        assert!(error.contains("interval_ms"), "{error}");
        assert!(error.contains("rtsp"), "{error}");
    }

    #[test]
    fn rejects_no_enabled_classes() {
        let text = SAMPLE
            .replace("detect_person = true", "detect_person = false")
            .replace("detect_car = true", "detect_car = false");
        let error = Config::from_toml(&text).unwrap_err().to_string();
        assert!(error.contains("at least one"), "{error}");
    }

    #[test]
    fn env_overrides_credentials_and_locks_camera() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        config
            .apply_env_overrides(EnvOverrides {
                username: Some("bob".into()),
                password: Some("newpass".into()),
                ..Default::default()
            })
            .unwrap();
        assert!(config.env_camera_locked);
        assert!(config.env_credentials_locked);
        assert_eq!(config.camera.username.as_deref(), Some("bob"));
        assert_eq!(config.camera.password.as_deref(), Some("newpass"));
        assert!(!config
            .effective_rtsp_url()
            .unwrap()
            .as_str()
            .contains("alice"));
    }

    #[test]
    fn update_rejects_camera_when_env_locked() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        config.env_camera_locked = true;
        let update = SettingsUpdate {
            camera: Some(CameraUpdate {
                rtsp_url: Some("rtsp://other.local/stream".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(config.apply_update(&update).is_err());
    }

    #[test]
    fn changing_host_clears_old_credentials() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        let update = SettingsUpdate {
            camera: Some(CameraUpdate {
                rtsp_url: Some("rtsp://192.168.1.101:554/stream1".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        config.apply_update(&update).unwrap();
        assert!(config.camera.username.is_none());
        assert!(config.camera.password.is_none());
    }

    #[test]
    fn url_pasted_into_settings_extracts_credentials() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        let update = SettingsUpdate {
            camera: Some(CameraUpdate {
                rtsp_url: Some("rtsp://carol:p%40ss@10.0.0.2:554/live".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        config.apply_update(&update).unwrap();
        assert_eq!(config.camera.username.as_deref(), Some("carol"));
        assert_eq!(config.camera.password.as_deref(), Some("p@ss"));
        assert!(!config.camera.rtsp_url.contains("carol"));
    }

    #[test]
    fn live_update_does_not_require_restart() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        let update = SettingsUpdate {
            yolo: Some(YoloUpdate {
                confidence_threshold: Some(0.7),
                interval_ms: Some(2_000),
                detect_car: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let outcome = config.apply_update(&update).unwrap();
        assert!(outcome.restart_required.is_empty());
        assert!(!outcome.camera_connection_changed);
        assert_eq!(config.yolo.confidence_threshold, 0.7);
        assert_eq!(config.yolo.interval_ms, 2_000);
        assert!(!config.yolo.detect_car);
    }

    #[test]
    fn model_change_requires_restart() {
        let mut config = Config::from_toml(SAMPLE).unwrap();
        let update = SettingsUpdate {
            yolo: Some(YoloUpdate {
                model_path: Some("models/other.onnx".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let outcome = config.apply_update(&update).unwrap();
        assert_eq!(outcome.restart_required, vec!["model_path"]);
    }

    #[test]
    fn debug_output_redacts_credentials() {
        let config = Config::from_toml(SAMPLE).unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains("alice"));
        assert!(!debug.contains("s3cret"));
        assert!(!debug.contains("s3cret@"));
    }
}
