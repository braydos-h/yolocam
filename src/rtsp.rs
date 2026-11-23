use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use bytes::{Buf, Bytes, BytesMut};
use memchr::memmem;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::config::CameraConfig;
use crate::state::{AppState, CameraStatus};

const MAX_FRAME_BYTES: usize = 24 * 1024 * 1024;
const MAX_BUFFER_BYTES: usize = MAX_FRAME_BYTES + (4 * 1024 * 1024);
const MAX_STDERR_LINES: usize = 20;
const RECONNECT_MAX_SECONDS: u64 = 60;
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
const JPEG_EOI: [u8; 2] = [0xFF, 0xD9];

/// Incremental parser for the concatenated JPEG stream produced by FFmpeg.
/// The internal buffer is bounded; corrupt or oversized data is discarded.
pub struct MjpegParser {
    buffer: BytesMut,
}

impl Default for MjpegParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MjpegParser {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(256 * 1024),
        }
    }

    pub fn push(&mut self, data: &[u8]) {
        if !data.is_empty() {
            if self.buffer.len().saturating_add(data.len()) > MAX_BUFFER_BYTES {
                tracing::warn!("MJPEG parser buffer overflow; discarding buffered bytes");
                self.buffer.clear();
            }
            self.buffer.extend_from_slice(data);
        }
    }

    pub fn next_frame(&mut self) -> Option<Bytes> {
        let Some(start) = memmem::find(&self.buffer, &JPEG_SOI) else {
            // Preserve a trailing 0xFF that may be the first half of SOI in
            // the next chunk; discard the rest of the garbage.
            let keep = usize::from(self.buffer.last() == Some(&0xFF));
            let len = self.buffer.len();
            if len > keep {
                self.buffer.advance(len - keep);
            }
            return None;
        };
        if start > 0 {
            self.buffer.advance(start);
        }
        if self.buffer.len() < 4 {
            return None;
        }
        let Some(relative_end) = memmem::find(&self.buffer[2..], &JPEG_EOI) else {
            if self.buffer.len() > MAX_FRAME_BYTES {
                tracing::warn!("discarding oversized MJPEG frame");
                self.buffer.clear();
            }
            return None;
        };
        let frame_len = 2 + relative_end + 2;
        Some(self.buffer.split_to(frame_len).freeze())
    }
}

pub fn ffmpeg_args(url: &Url, preview_width: u32) -> Vec<String> {
    vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "warning".to_string(),
        "-nostdin".to_string(),
        "-rtsp_transport".to_string(),
        "tcp".to_string(),
        "-rw_timeout".to_string(),
        "15000000".to_string(),
        "-analyzeduration".to_string(),
        "5000000".to_string(),
        "-probesize".to_string(),
        "5000000".to_string(),
        "-i".to_string(),
        url.to_string(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-an".to_string(),
        "-vf".to_string(),
        format!("scale=w=min({preview_width}\\,iw):h=-2"),
        "-f".to_string(),
        "mjpeg".to_string(),
        "-q:v".to_string(),
        "5".to_string(),
        "-fps_mode".to_string(),
        "passthrough".to_string(),
        "pipe:1".to_string(),
    ]
}

pub fn redact_secrets(text: &str, secrets: &[String]) -> String {
    let mut output = text.to_string();
    let mut secrets = secrets.to_vec();
    secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
    for secret in secrets {
        if !secret.is_empty() {
            output = output.replace(&secret, "***");
        }
    }
    output
}

pub async fn probe_ffmpeg(path: &str) -> Result<String, String> {
    let mut command = Command::new(path);
    command.arg("-version");
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    configure_command(&mut command);
    let output = timeout(Duration::from_secs(8), command.output())
        .await
        .map_err(|_| "FFmpeg did not respond within 8 seconds".to_string())?
        .map_err(|error| format!("cannot execute FFmpeg ({path}): {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "FFmpeg exited with status {} while checking version",
            output.status
        ));
    }
    let version = String::from_utf8_lossy(&output.stdout);
    Ok(version
        .lines()
        .next()
        .unwrap_or("ffmpeg version unknown")
        .trim()
        .to_string())
}

#[derive(Clone)]
pub struct CameraManager {
    running: Arc<Mutex<Option<CameraRun>>>,
}

struct CameraRun {
    token: CancellationToken,
    handle: JoinHandle<()>,
}

impl CameraManager {
    pub fn new() -> Self {
        Self {
            running: Arc::new(Mutex::new(None)),
        }
    }

    /// Start the camera if it is not already running. Returns true when a new
    /// worker was spawned.
    pub async fn start(&self, app: AppState) -> bool {
        let mut running = self.running.lock().await;
        if let Some(existing) = running.as_ref() {
            if !existing.handle.is_finished() {
                return false;
            }
        }
        running.take();
        app.frame_store().clear();
        app.clear_analysis();
        app.set_camera_status(CameraStatus::connecting());

        let token = CancellationToken::new();
        let worker_token = token.clone();
        let worker_app = app.clone();
        let handle = tokio::spawn(async move {
            run_camera_worker(worker_app, worker_token).await;
        });
        *running = Some(CameraRun { token, handle });
        true
    }

    pub async fn stop(&self, app: &AppState) {
        let run = self.running.lock().await.take();
        if let Some(run) = run {
            run.token.cancel();
            let mut handle = run.handle;
            match timeout(Duration::from_secs(10), &mut handle).await {
                Ok(_) => {}
                Err(_) => {
                    tracing::warn!("camera worker did not stop within 10 seconds; aborting task");
                    handle.abort();
                }
            }
        }
        app.frame_store().clear();
        app.clear_analysis();
        app.set_camera_status(CameraStatus::stopped());
    }

    pub async fn reconnect(&self, app: AppState) {
        self.stop(&app).await;
        self.start(app).await;
    }

    pub async fn is_running(&self) -> bool {
        let running = self.running.lock().await;
        running
            .as_ref()
            .is_some_and(|run| !run.handle.is_finished())
    }
}

impl Default for CameraManager {
    fn default() -> Self {
        Self::new()
    }
}

enum StreamEnd {
    Cancelled,
    Failed { message: String, had_frames: bool },
}

async fn run_camera_worker(app: AppState, token: CancellationToken) {
    let mut attempt: u32 = 0;
    loop {
        if token.is_cancelled() || app.is_shutting_down() {
            break;
        }
        let config = app.config();
        let delay = config.camera.reconnect_delay_seconds.max(1);
        let effective_url = match config.effective_rtsp_url() {
            Ok(url) => url,
            Err(error) => {
                app.set_camera_status(CameraStatus::error(format!(
                    "Invalid RTSP configuration: {error}"
                )));
                break;
            }
        };
        let secrets = config.secrets();

        match stream_until_failure(&app, &token, &config.camera, &effective_url, &secrets).await {
            StreamEnd::Cancelled => break,
            StreamEnd::Failed {
                message,
                had_frames,
            } => {
                if token.is_cancelled() || app.is_shutting_down() {
                    break;
                }
                if had_frames {
                    attempt = 0;
                }
                attempt = attempt.saturating_add(1);
                let wait_seconds = reconnect_delay(delay, attempt);
                app.clear_analysis();
                app.set_camera_status(CameraStatus::reconnecting(
                    attempt,
                    wait_seconds,
                    message.clone(),
                ));
                tracing::warn!(
                    attempt,
                    wait_seconds,
                    error = %message,
                    "RTSP stream stopped; scheduling reconnect"
                );
                tokio::select! {
                    _ = token.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(wait_seconds)) => {}
                }
            }
        }
    }
}

fn reconnect_delay(base_seconds: u64, attempt: u32) -> u64 {
    let exponent = attempt.saturating_sub(1).min(4);
    base_seconds
        .max(1)
        .saturating_mul(1u64 << exponent)
        .min(RECONNECT_MAX_SECONDS)
}

async fn stream_until_failure(
    app: &AppState,
    token: &CancellationToken,
    camera: &CameraConfig,
    effective_url: &Url,
    secrets: &[String],
) -> StreamEnd {
    let mut command = Command::new(&camera.ffmpeg_path);
    command.args(ffmpeg_args(effective_url, camera_preview_width(app)));
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    command.kill_on_drop(true);
    configure_command(&mut command);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            return StreamEnd::Failed {
                message: format!(
                    "Unable to start FFmpeg ({}) : {error}",
                    camera.ffmpeg_path
                ),
                had_frames: false,
            };
        }
    };

    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            terminate_child(&mut child).await;
            return StreamEnd::Failed {
                message: "FFmpeg stdout was not available".to_string(),
                had_frames: false,
            };
        }
    };
    let stderr_task = child.stderr.take().map(|stderr| {
        let secrets = secrets.to_vec();
        tokio::spawn(read_stderr(stderr, secrets))
    });

    let mut stdout = stdout;
    let mut parser = MjpegParser::new();
    let mut read_buffer = BytesMut::with_capacity(256 * 1024);
    let mut had_frames = false;

    loop {
        tokio::select! {
            _ = token.cancelled() => {
                terminate_child(&mut child).await;
                if let Some(task) = stderr_task {
                    task.abort();
                }
                return StreamEnd::Cancelled;
            }
            read = stdout.read_buf(&mut read_buffer) => {
                match read {
                    Ok(0) => {
                        let status = child.wait().await;
                        let stderr_lines = collect_stderr(stderr_task).await;
                        let detail = last_stderr_line(&stderr_lines)
                            .unwrap_or_else(|| "FFmpeg exited without an error message".to_string());
                        let code = status
                            .as_ref()
                            .map(|status| status.to_string())
                            .unwrap_or_else(|error| format!("wait failed: {error}"));
                        return StreamEnd::Failed {
                            message: redact_secrets(
                                &format!("FFmpeg stream ended ({code}): {detail}"),
                                secrets,
                            ),
                            had_frames,
                        };
                    }
                    Ok(n) => {
                        parser.push(&read_buffer[..n]);
                        read_buffer.clear();
                        while let Some(frame) = parser.next_frame() {
                            app.frame_store().publish(frame);
                            if !had_frames {
                                had_frames = true;
                                app.set_camera_status(CameraStatus::streaming());
                            }
                        }
                    }
                    Err(error) => {
                        terminate_child(&mut child).await;
                        let stderr_lines = collect_stderr(stderr_task).await;
                        let detail = last_stderr_line(&stderr_lines)
                            .unwrap_or_else(|| error.to_string());
                        return StreamEnd::Failed {
                            message: redact_secrets(
                                &format!("FFmpeg output read failed: {detail}"),
                                secrets,
                            ),
                            had_frames,
                        };
                    }
                }
            }
        }
    }
}

fn camera_preview_width(app: &AppState) -> u32 {
    app.config().preview_width()
}

async fn read_stderr(
    stderr: tokio::process::ChildStderr,
    secrets: Vec<String>,
) -> VecDeque<String> {
    let mut lines = VecDeque::with_capacity(MAX_STDERR_LINES);
    let mut reader = BufReader::new(stderr).lines();
    loop {
        match reader.next_line().await {
            Ok(Some(line)) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if lines.len() == MAX_STDERR_LINES {
                    lines.pop_front();
                }
                lines.push_back(redact_secrets(&line, &secrets));
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    lines
}

async fn collect_stderr(
    task: Option<JoinHandle<VecDeque<String>>>,
) -> VecDeque<String> {
    let Some(task) = task else {
        return VecDeque::new();
    };
    match timeout(Duration::from_secs(2), task).await {
        Ok(Ok(lines)) => lines,
        Ok(Err(_)) => VecDeque::new(),
        Err(_) => VecDeque::new(),
    }
}

fn last_stderr_line(lines: &VecDeque<String>) -> Option<String> {
    lines
        .iter()
        .rev()
        .find(|line| !line.trim().is_empty())
        .cloned()
}

async fn terminate_child(child: &mut Child) {
    if let Err(error) = child.start_kill() {
        tracing::debug!(error = %error, "failed to signal FFmpeg process");
    }
    let _ = timeout(Duration::from_secs(3), child.wait()).await;
}

fn configure_command(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW; avoids a console window when launched by the
        // dashboard/host process.
        command.as_std_mut().creation_flags(0x0800_0000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_jpeg(payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&JPEG_SOI);
        bytes.extend_from_slice(b"fake jpeg payload");
        bytes.extend_from_slice(payload);
        bytes.extend_from_slice(&JPEG_EOI);
        bytes
    }

    #[test]
    fn parser_extracts_multiple_frames_and_skips_garbage() {
        let first = build_jpeg(b"one");
        let second = build_jpeg(b"two");
        let mut input = Vec::new();
        input.extend_from_slice(b"garbage before first frame");
        input.extend_from_slice(&first);
        input.extend_from_slice(&second);

        let mut parser = MjpegParser::new();
        parser.push(&input);
        let parsed_first = parser.next_frame().unwrap();
        assert!(parsed_first.starts_with(&JPEG_SOI));
        assert!(parsed_first.ends_with(&JPEG_EOI));
        assert_eq!(&parsed_first[..], &first[..]);
        let parsed_second = parser.next_frame().unwrap();
        assert_eq!(&parsed_second[..], &second[..]);
        assert!(parser.next_frame().is_none());
    }

    #[test]
    fn parser_handles_markers_split_across_chunks() {
        let frame = build_jpeg(b"split");
        let mut parser = MjpegParser::new();
        for chunk in frame.chunks(3) {
            parser.push(chunk);
        }
        let parsed = parser.next_frame().unwrap();
        assert_eq!(&parsed[..], &frame[..]);
    }

    #[test]
    fn parser_keeps_trailing_ff_and_bounds_garbage() {
        let mut parser = MjpegParser::new();
        parser.push(&[0x00, 0x01, 0xFF]);
        assert_eq!(&parser.buffer[..], &[0xFF]);
        parser.push(&[0xD8, 0x00, 0xFF, 0xD9]);
        let frame = parser.next_frame().unwrap();
        assert_eq!(&frame[..], &[0xFF, 0xD8, 0x00, 0xFF, 0xD9]);
    }

    #[test]
    fn ffmpeg_args_use_tcp_and_stream_mjpeg() {
        let url = Url::parse("rtsp://user:pass@example.test:554/stream").unwrap();
        let args = ffmpeg_args(&url, 1280);
        let joined = args.join(" ");
        assert!(joined.contains("-rtsp_transport tcp"));
        assert!(joined.contains("-f mjpeg"));
        assert!(joined.contains("pipe:1"));
        assert!(args.iter().any(|arg| arg.contains("min(1280")));
    }

    #[test]
    fn redaction_removes_url_and_credentials() {
        let secrets = vec![
            "rtsp://user:pass@example.test/stream".to_string(),
            "user".to_string(),
            "pass".to_string(),
        ];
        let message = "401 Unauthorized for rtsp://user:pass@example.test/stream by user using pass";
        let redacted = redact_secrets(message, &secrets);
        assert!(!redacted.contains("user"));
        assert!(!redacted.contains("pass"));
        assert!(redacted.contains("***"));
    }

    #[test]
    fn reconnect_delay_backs_off_and_caps() {
        assert_eq!(reconnect_delay(5, 1), 5);
        assert_eq!(reconnect_delay(5, 2), 10);
        assert_eq!(reconnect_delay(5, 3), 20);
        assert_eq!(reconnect_delay(5, 4), 40);
        assert_eq!(reconnect_delay(5, 5), 60);
        assert_eq!(reconnect_delay(5, 20), 60);
    }
}
