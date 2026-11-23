use std::convert::Infallible;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;

use crate::state::AppState;

const STREAM_FPS: u64 = 15;
const BOUNDARY: &str = "frame";

/// Browser-compatible MJPEG stream. Only the newest JPEG is kept by the frame
/// store, so a slow client falls behind by skipping frames rather than using
/// unbounded memory.
pub async fn mjpeg_stream(State(app): State<AppState>) -> Response {
    let stream = async_stream::stream! {
        let mut receiver = app.frame_store().subscribe();
        let mut shutdown = app.shutdown_token();
        let mut last_sent: Option<Instant> = None;

        if let Some(frame) = receiver.borrow().clone() {
            if let Some(header) = encode_part(&frame.jpeg, &mut last_sent) {
                yield Ok::<Bytes, Infallible>(header);
                yield Ok(frame.jpeg.clone());
                yield Ok(Bytes::from_static(b"\r\n"));
            }
        }

        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                changed = receiver.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    let frame = receiver.borrow_and_update().clone();
                    let Some(frame) = frame else {
                        continue;
                    };
                    if let Some(header) = encode_part(&frame.jpeg, &mut last_sent) {
                        yield Ok(header);
                        yield Ok(frame.jpeg.clone());
                        yield Ok(Bytes::from_static(b"\r\n"));
                    }
                }
            }
        }
    };

    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("multipart/x-mixed-replace; boundary=frame"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store, no-cache"));
    response.headers_mut().insert(
        header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    response.headers_mut().insert(
        header::HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
    response
}

/// Also yield the JPEG body itself. Kept separate from the header to avoid
/// copying the JPEG bytes on every client.
fn encode_part(jpeg: &Bytes, last_sent: &mut Option<Instant>) -> Option<Bytes> {
    let now = Instant::now();
    if let Some(last) = *last_sent {
        if now.duration_since(last) < Duration::from_millis(1_000 / STREAM_FPS) {
            return None;
        }
    }
    *last_sent = Some(now);
    Some(Bytes::from(mjpeg_header(jpeg.len())))
}

fn mjpeg_header(content_length: usize) -> Vec<u8> {
    format!(
        "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {content_length}\r\n\r\n"
    )
    .into_bytes()
}

/// Also yield the JPEG body itself. Kept separate from the header to avoid
/// copying the JPEG bytes on every client.
async fn yield_frame() {}

pub async fn latest_jpeg(State(app): State<AppState>) -> Response {
    match app.frame_store().latest() {
        Some(frame) => {
            let mut response = Response::new(Body::from(frame.jpeg.clone()));
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("image/jpeg"),
            );
            response.headers_mut().insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("no-store, no-cache"),
            );
            response
        }
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mjpeg_header_is_well_formed() {
        let header = String::from_utf8(mjpeg_header(1234)).unwrap();
        assert!(header.starts_with("--frame\r\n"));
        assert!(header.contains("Content-Type: image/jpeg\r\n"));
        assert!(header.contains("Content-Length: 1234\r\n"));
        assert!(header.ends_with("\r\n\r\n"));
    }
}
