//! Integration test for the simulated video input path:
//! concatenated MJPEG chunks -> bounded parser -> latest-frame store ->
//! once-per-interval sampling. No camera, FFmpeg, or ONNX model is required.

use std::time::Duration;

use bytes::Bytes;
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, Rgb, RgbImage};
use yolocam::frame::{next_analysis_deadline, sample_latest, FrameStore};
use yolocam::rtsp::MjpegParser;

fn make_jpeg(width: u32, height: u32, color: [u8; 3]) -> Vec<u8> {
    let image = RgbImage::from_pixel(width, height, Rgb(color));
    let mut bytes = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut bytes, 80);
    encoder
        .encode(image.as_raw(), width, height, ExtendedColorType::Rgb8)
        .expect("encode test JPEG");
    bytes
}

#[test]
fn simulated_video_stream_uses_only_the_latest_frame() {
    let frames = [
        make_jpeg(320, 180, [200, 30, 30]),
        make_jpeg(320, 180, [30, 200, 30]),
        make_jpeg(320, 180, [30, 30, 200]),
    ];

    // Simulate network/host chunk boundaries, including SOI/EOI markers split
    // across reads and unrelated bytes between JPEG frames.
    let mut stream = Vec::new();
    stream.extend_from_slice(b"\x00\x01transport noise");
    for (index, frame) in frames.iter().enumerate() {
        stream.extend_from_slice(frame);
        if index + 1 < frames.len() {
            stream.extend_from_slice(b"\x00\x00media boundary\x00");
        }
    }

    let mut parser = MjpegParser::new();
    let store = FrameStore::new();
    let mut parsed = 0usize;
    for chunk in stream.chunks(17) {
        parser.push(chunk);
        while let Some(frame) = parser.next_frame() {
            assert!(frame.starts_with(&[0xFF, 0xD8]));
            assert!(frame.ends_with(&[0xFF, 0xD9]));
            store.publish(frame);
            parsed += 1;
        }
    }

    assert_eq!(parsed, frames.len());
    assert_eq!(store.sequence(), frames.len() as u64);
    let latest = store.latest().expect("latest frame");
    assert_eq!(latest.sequence, frames.len() as u64);

    // The inference scheduler sampled once per second: at each tick it only
    // sees the newest frame and never replays older frames.
    let after = sample_latest(&store, 1).expect("frame newer than sequence 1");
    assert_eq!(after.sequence, 3);

    let first = store.stats();
    assert_eq!(first.frames_received, 3);
    assert!(first.last_frame_at.is_some());
}

#[test]
fn sampling_schedule_skips_missed_intervals() {
    let start = std::time::Instant::now();
    let interval = Duration::from_secs(1);
    // A slow inference run ends well past the next deadline; the schedule
    // should target "now" rather than a backlog of stale 1-second ticks.
    let next = next_analysis_deadline(start, interval, start + Duration::from_secs(4));
    assert_eq!(next, start + Duration::from_secs(4));
}

#[test]
fn stale_frames_are_not_processed_twice() {
    let store = FrameStore::new();
    let frame = store.publish(Bytes::from_static(b"jpeg"));
    assert!(sample_latest(&store, frame.sequence).is_none());
    let next = store.publish(Bytes::from_static(b"jpeg2"));
    assert_eq!(
        sample_latest(&store, frame.sequence).unwrap().sequence,
        next.sequence
    );
}
