use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::DateTime;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::broadcast;

use crate::config::{ConfigError, SettingsUpdate};
use crate::database::HistoryFilter;
use crate::detection::DetectedClass;
use crate::error::ApiError;
use crate::state::{AppState, CameraStatus, WsEvent};

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_CSS: &str = include_str!("../web/app.css");
const APP_JS: &str = include_str!("../web/app.js");

const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/api/health", get(health))
        .route("/api/stats", get(stats))
        .route(
            "/api/settings",
            get(get_settings).post(update_settings).put(update_settings),
        )
        .route("/api/history", get(history))
        .route("/api/stream", get(crate::stream::mjpeg_stream))
        .route("/api/frame.jpg", get(crate::stream::latest_jpeg))
        .route("/api/screenshots/{filename}", get(screenshot))
        .route("/api/camera/start", post(camera_start))
        .route("/api/camera/stop", post(camera_stop))
        .route("/api/camera/reconnect", post(camera_reconnect))
        .route("/api/hardware", get(hardware))
        .route("/api/hardware/benchmark", post(run_hardware_benchmark))
        .route("/api/discover", get(discover_cameras))
        .route("/ws", get(websocket))
        .fallback(not_found)
        .with_state(state)
        .layer(axum::middleware::from_fn(security_headers))
}

async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
    );
    response
}

async fn index() -> Response {
    static_asset(INDEX_HTML, "text/html; charset=utf-8")
}

async fn css() -> Response {
    static_asset(APP_CSS, "text/css; charset=utf-8")
}

async fn js() -> Response {
    static_asset(APP_JS, "text/javascript; charset=utf-8")
}

fn static_asset(content: &'static str, mime: &'static str) -> Response {
    let mut response = content.into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

async fn not_found() -> ApiError {
    ApiError::not_found("not found")
}

async fn health(State(app): State<AppState>) -> Json<serde_json::Value> {
    let snapshot = app.snapshot();
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "camera": snapshot.camera.status,
        "ai": snapshot.ai.state,
        "uptime_seconds": snapshot.uptime_seconds,
    }))
}

async fn stats(State(app): State<AppState>) -> Json<crate::state::LiveSnapshot> {
    Json(app.snapshot())
}

#[derive(Debug, Deserialize)]
struct DiscoverQuery {
    subnet: Option<String>,
    timeout_ms: Option<u64>,
}

async fn discover_cameras(
    Query(query): Query<DiscoverQuery>,
) -> Result<Json<crate::discovery::DiscoveryReport>, ApiError> {
    let report = crate::discovery::discover(query.subnet.as_deref(), query.timeout_ms.unwrap_or(450))
        .await
        .map_err(|error| match error {
            crate::discovery::DiscoveryError::Busy => ApiError::conflict(error.to_string()),
            crate::discovery::DiscoveryError::Invalid(_) => ApiError::bad_request(error.to_string()),
            crate::discovery::DiscoveryError::Internal(_) => ApiError::internal(error.to_string()),
        })?;
    tracing::info!(
        hosts = report.hosts_scanned,
        streams = report.streams.len(),
        subnets = ?report.scanned_subnets,
        "camera discovery finished"
    );
    Ok(Json(report))
}

async fn hardware(State(app): State<AppState>) -> Json<serde_json::Value> {
    let config = app.config();
    let ai = app.ai_status();
    let enabled = config.yolo.model_path == crate::config::AUTO_MODEL_PATH;
    Json(json!({
        "enabled": enabled,
        "config": config.yolo.auto,
        "target_ms": crate::hardware::target_ms(&config.yolo),
        "hardware": crate::hardware::describe(config.yolo.device),
        "candidates": crate::models::public_candidates(config.yolo.auto.max_model),
        "selection": ai.auto,
        "state": ai.state,
        "model_path": ai.model_path,
    }))
}

async fn run_hardware_benchmark(
    State(app): State<AppState>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    let config = app.config();
    if config.yolo.model_path != crate::config::AUTO_MODEL_PATH {
        return Err(ApiError::bad_request(
            "automatic model selection is disabled; set yolo.model_path = \"auto\" first",
        ));
    }
    if app.ai_status().state == crate::state::AiState::Benchmarking {
        return Err(ApiError::conflict("the hardware test is already running"));
    }
    if !app.request_rebenchmark() {
        return Err(ApiError::service_unavailable(
            "the inference worker is not available",
        ));
    }
    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "started" }))))
}

async fn get_settings(State(app): State<AppState>) -> Json<crate::config::PublicSettings> {
    Json(app.config().public_settings())
}

async fn update_settings(
    State(app): State<AppState>,
    Json(update): Json<SettingsUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let current = app.config();
    let mut candidate = current.clone();
    let outcome = candidate.apply_update(&update)?;
    candidate.validate()?;

    candidate
        .save(app.config_path())
        .map_err(|error| match error {
            ConfigError::Io(io_error) => ApiError::internal(format!(
                "settings validated but could not be saved: {io_error}"
            )),
            other => ApiError::bad_request(other.to_string()),
        })?;

    let env_locked = candidate.env_camera_locked;
    app.replace_config(candidate.clone());

    let was_running = app.camera().is_running().await;
    let mut camera_reconnecting = false;
    if outcome.camera_connection_changed {
        if was_running {
            app.camera().reconnect(app.clone()).await;
            camera_reconnecting = true;
        } else {
            app.set_camera_status(CameraStatus::stopped());
        }
    } else {
        app.publish_snapshot();
    }

    Ok(Json(json!({
        "settings": candidate.public_settings(),
        "restart_required": outcome.restart_required,
        "camera_reconnecting": camera_reconnecting,
        "environment_locked": env_locked,
    })))
}

#[derive(Debug, Deserialize)]
struct HistoryQuery {
    class: Option<String>,
    from: Option<String>,
    to: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
}

async fn history(
    State(app): State<AppState>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let class = match query
        .class
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(value) => Some(
            DetectedClass::from_name(value)
                .ok_or_else(|| ApiError::bad_request("class must be person or car"))?,
        ),
        None => None,
    };
    let from_epoch_ms = query
        .from
        .as_deref()
        .map(|value| parse_timestamp(value, "from"))
        .transpose()?;
    let to_epoch_ms = query
        .to
        .as_deref()
        .map(|value| parse_timestamp(value, "to"))
        .transpose()?;
    if let (Some(from), Some(to)) = (from_epoch_ms, to_epoch_ms) {
        if from > to {
            return Err(ApiError::bad_request("from must be before to"));
        }
    }
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let offset = query.offset.unwrap_or(0).min(1_000_000);
    let records = app
        .database()
        .history(HistoryFilter {
            class,
            from_epoch_ms,
            to_epoch_ms,
            limit,
            offset,
        })
        .await
        .map_err(|error| ApiError::internal(format!("history query failed: {error}")))?;

    Ok(Json(json!({
        "items": records,
        "limit": limit,
        "offset": offset,
    })))
}

fn parse_timestamp(value: &str, name: &str) -> Result<i64, ApiError> {
    DateTime::parse_from_rfc3339(value.trim())
        .map(|timestamp| timestamp.timestamp_millis())
        .map_err(|_| ApiError::bad_request(format!("{name} must be an RFC 3339 timestamp")))
}

async fn screenshot(
    State(app): State<AppState>,
    AxumPath(filename): AxumPath<String>,
) -> Result<Response, ApiError> {
    if filename.is_empty()
        || filename.len() > 160
        || filename.starts_with('.')
        || filename.contains("..")
        || filename.contains('/')
        || filename.contains('\\')
        || !filename.to_ascii_lowercase().ends_with(".jpg")
        || !filename.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return Err(ApiError::bad_request("invalid screenshot filename"));
    }
    let directory = app.config().screenshot_dir();
    let path = directory.join(&filename);
    if Path::new(&filename)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(filename.as_str())
    {
        return Err(ApiError::bad_request("invalid screenshot filename"));
    }
    let bytes = tokio::fs::read(&path).await.map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ApiError::not_found("screenshot not found")
        } else {
            ApiError::internal(format!("cannot read screenshot: {error}"))
        }
    })?;
    let mut response = Response::new(bytes.into());
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("image/jpeg"));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    Ok(response)
}

async fn camera_start(State(app): State<AppState>) -> Json<serde_json::Value> {
    let started = app.camera().start(app.clone()).await;
    Json(json!({
        "started": started,
        "running": app.camera().is_running().await,
        "snapshot": app.snapshot(),
    }))
}

async fn camera_stop(State(app): State<AppState>) -> Json<serde_json::Value> {
    app.camera().stop(&app).await;
    Json(json!({
        "running": false,
        "snapshot": app.snapshot(),
    }))
}

async fn camera_reconnect(State(app): State<AppState>) -> Json<serde_json::Value> {
    app.camera().reconnect(app.clone()).await;
    Json(json!({
        "running": app.camera().is_running().await,
        "snapshot": app.snapshot(),
    }))
}

async fn websocket(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn handle_socket(socket: WebSocket, app: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let mut events = app.subscribe_events();
    let shutdown = app.shutdown_token();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(1));
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    if send_snapshot(&mut sender, &app).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => {
                let _ = sender.send(Message::Close(None)).await;
                break;
            }
            event = events.recv() => {
                match event {
                    Ok(WsEvent::Snapshot(snapshot)) => {
                        if send_json(&mut sender, &WsEvent::Snapshot(snapshot)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // The client missed intermediate frames; the latest
                        // state is enough for this dashboard.
                        if send_snapshot(&mut sender, &app).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            incoming = receiver.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(payload))) => {
                        if sender.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
            _ = heartbeat.tick() => {
                // Keep FPS/last-frame values fresh even when there are no new
                // AI results (which arrive only once per detection interval).
                if send_snapshot(&mut sender, &app).await.is_err() {
                    break;
                }
            }
        }
    }
}

async fn send_snapshot(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    app: &AppState,
) -> Result<(), ()> {
    let snapshot = app.snapshot();
    send_json(sender, &WsEvent::Snapshot(Box::new(snapshot))).await
}

async fn send_json(
    sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
    event: &WsEvent,
) -> Result<(), ()> {
    let json = serde_json::to_string(event).map_err(|error| {
        tracing::error!(error = %error, "failed to serialize WebSocket event");
    })?;
    sender
        .send(Message::Text(json.into()))
        .await
        .map_err(|_| ())
}

/// Keep the state type visible to rustdoc and allow external callers to
/// construct the built-in assets without duplicating paths.
pub fn embedded_assets() -> [(&'static str, &'static str); 3] {
    [
        ("index.html", INDEX_HTML),
        ("app.css", APP_CSS),
        ("app.js", APP_JS),
    ]
}

#[allow(dead_code)]
fn _assert_state_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AppState>();
    assert_send_sync::<Arc<AppState>>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_parser_rejects_garbage() {
        assert!(parse_timestamp("yesterday", "from").is_err());
        assert!(parse_timestamp("2026-11-23T19:30:01Z", "from").is_ok());
    }

    #[test]
    fn embedded_assets_are_present() {
        for (name, content) in embedded_assets() {
            assert!(!content.trim().is_empty(), "{name} is empty");
        }
    }
}
