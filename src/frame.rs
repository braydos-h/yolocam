use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use tokio::sync::watch;

/// A decoded JPEG preview frame plus the metadata sampling needs.
#[derive(Debug)]
pub struct VideoFrame {
    pub sequence: u64,
    pub jpeg: Bytes,
    pub captured_at: Instant,
    pub epoch_ms: i64,
}

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub sequence: u64,
    pub camera_fps: f32,
    pub last_frame_at: Option<String>,
    pub frames_received: u64,
}

#[derive(Clone)]
pub struct FrameStore {
    inner: Arc<FrameStoreInner>,
}

struct FrameStoreInner {
    tx: watch::Sender<Option<Arc<VideoFrame>>>,
    sequence: AtomicU64,
    frames_received: AtomicU64,
    camera_fps_bits: AtomicU32,
    last_frame_epoch_ms: AtomicI64,
    fps_meter: Mutex<FpsMeter>,
}

impl Default for FrameStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStore {
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(None);
        Self {
            inner: Arc::new(FrameStoreInner {
                tx,
                sequence: AtomicU64::new(0),
                frames_received: AtomicU64::new(0),
                camera_fps_bits: AtomicU32::new(0f32.to_bits()),
                last_frame_epoch_ms: AtomicI64::new(0),
                fps_meter: Mutex::new(FpsMeter::new(Duration::from_secs(2))),
            }),
        }
    }

    /// Replace the latest frame. Watch channels keep only the newest value, so
    /// slow consumers can never grow an unbounded queue.
    pub fn publish(&self, jpeg: Bytes) -> Arc<VideoFrame> {
        let now = Instant::now();
        let epoch = Utc::now();
        let sequence = self.inner.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let fps = {
            let mut meter = self
                .inner
                .fps_meter
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            meter.record_at(now);
            meter.fps_at(now)
        };
        self.inner
            .camera_fps_bits
            .store(fps.to_bits(), Ordering::Relaxed);
        self.inner
            .frames_received
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .last_frame_epoch_ms
            .store(epoch.timestamp_millis(), Ordering::Relaxed);

        let frame = Arc::new(VideoFrame {
            sequence,
            jpeg,
            captured_at: now,
            epoch_ms: epoch.timestamp_millis(),
        });
        self.inner.tx.send_replace(Some(Arc::clone(&frame)));
        frame
    }

    pub fn latest(&self) -> Option<Arc<VideoFrame>> {
        self.inner.tx.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<Option<Arc<VideoFrame>>> {
        self.inner.tx.subscribe()
    }

    pub fn sequence(&self) -> u64 {
        self.inner.sequence.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> FrameStats {
        let last_epoch_ms = self.inner.last_frame_epoch_ms.load(Ordering::Relaxed);
        FrameStats {
            sequence: self.sequence(),
            camera_fps: f32::from_bits(self.inner.camera_fps_bits.load(Ordering::Relaxed)),
            last_frame_at: if last_epoch_ms > 0 {
                DateTime::<Utc>::from_timestamp_millis(last_epoch_ms).map(|timestamp| {
                    timestamp.to_rfc3339_opts(SecondsFormat::Millis, true)
                })
            } else {
                None
            },
            frames_received: self.inner.frames_received.load(Ordering::Relaxed),
        }
    }

    /// Clear the displayed frame, e.g. when the camera is stopped.
    pub fn clear(&self) {
        self.inner
            .camera_fps_bits
            .store(0f32.to_bits(), Ordering::Relaxed);
        self.inner.last_frame_epoch_ms.store(0, Ordering::Relaxed);
        {
            let mut meter = self
                .inner
                .fps_meter
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *meter = FpsMeter::new(Duration::from_secs(2));
        }
        self.inner.tx.send_replace(None);
    }
}

/// Sliding-window frame rate meter. Timestamps are injected so the logic can
/// be unit tested without sleeping.
pub struct FpsMeter {
    window: Duration,
    samples: VecDeque<Instant>,
    max_samples: usize,
}

impl FpsMeter {
    pub fn new(window: Duration) -> Self {
        Self {
            window,
            samples: VecDeque::with_capacity(256),
            max_samples: 1_000,
        }
    }

    pub fn record_at(&mut self, now: Instant) {
        self.samples.push_back(now);
        self.prune(now);
        while self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    pub fn fps_at(&mut self, now: Instant) -> f32 {
        self.prune(now);
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first = *self.samples.front().expect("length checked");
        let last = *self.samples.back().expect("length checked");
        let elapsed = last.saturating_duration_since(first).as_secs_f32();
        if elapsed < 0.05 {
            return 0.0;
        }
        ((self.samples.len() - 1) as f32 / elapsed).min(1_000.0)
    }

    fn prune(&mut self, now: Instant) {
        let cutoff = now.checked_sub(self.window).unwrap_or(now);
        while let Some(front) = self.samples.front() {
            if *front < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Schedule the next analysis. If inference overran its interval, run at
/// `now` instead of accumulating a backlog of stale deadlines.
pub fn next_analysis_deadline(previous_due: Instant, interval: Duration, now: Instant) -> Instant {
    let next = previous_due + interval;
    if next <= now {
        now
    } else {
        next
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SamplingInfo {
    pub sequence: u64,
    pub delay_ms: u64,
}

/// Small helper used by tests to prove that sampling picks the newest frame.
pub fn sample_latest(store: &FrameStore, after_sequence: u64) -> Option<Arc<VideoFrame>> {
    store
        .latest()
        .filter(|frame| frame.sequence > after_sequence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_frame_replaces_previous_frame() {
        let store = FrameStore::new();
        assert!(store.latest().is_none());
        let first = store.publish(Bytes::from_static(b"first"));
        let second = store.publish(Bytes::from_static(b"second"));
        assert_eq!(second.sequence, first.sequence + 1);
        assert_eq!(store.latest().unwrap().jpeg, Bytes::from_static(b"second"));
        assert_eq!(store.stats().frames_received, 2);
        store.clear();
        assert!(store.latest().is_none());
        assert_eq!(store.stats().camera_fps, 0.0);
    }

    #[test]
    fn fps_meter_uses_sliding_window() {
        let start = Instant::now();
        let mut meter = FpsMeter::new(Duration::from_secs(2));
        for i in 0..21 {
            meter.record_at(start + Duration::from_millis(i * 50));
        }
        let fps = meter.fps_at(start + Duration::from_secs(1));
        assert!((fps - 20.0).abs() < 0.5, "fps was {fps}");
    }

    #[test]
    fn deadline_skips_stale_intervals() {
        let start = Instant::now();
        let interval = Duration::from_secs(1);
        let on_time = next_analysis_deadline(start, interval, start + Duration::from_millis(200));
        assert_eq!(on_time, start + interval);
        let late = next_analysis_deadline(start, interval, start + Duration::from_secs(3));
        assert_eq!(late, start + Duration::from_secs(3));
    }

    #[test]
    fn sampling_returns_only_newer_frames() {
        let store = FrameStore::new();
        let first = store.publish(Bytes::from_static(b"a"));
        assert!(sample_latest(&store, first.sequence).is_none());
        let second = store.publish(Bytes::from_static(b"b"));
        let sampled = sample_latest(&store, first.sequence).unwrap();
        assert_eq!(sampled.sequence, second.sequence);
    }
}
