//! Optional real-model integration test.
//!
//! This test is ignored by default because it needs the ONNX Runtime shared
//! library and a YOLO11/YOLOv8 COCO ONNX model. Run it locally with:
//!
//!   set YOLO_TEST_MODEL=models\yolo11n.onnx
//!   cargo test --test yolo_onnx -- --ignored --nocapture
//!
//! On Linux use `YOLO_TEST_MODEL=models/yolo11n.onnx cargo test ...`.
//! The ONNX Runtime library is auto-detected from the working directory or
//! ORT_DYLIB_PATH / YOLOCAM_ONNXRUNTIME_PATH.

use image::{Rgb, RgbImage};
use yolocam::config::{InferenceDevice, YoloConfig};
use yolocam::detection::ClassFilter;
use yolocam::yolo::YoloEngine;

#[test]
#[ignore = "requires an ONNX model and the ONNX Runtime shared library"]
fn model_loads_and_runs_on_a_generated_frame() {
    let model_path =
        std::env::var("YOLO_TEST_MODEL").unwrap_or_else(|_| "models/yolo11n.onnx".to_string());
    if !std::path::Path::new(&model_path).is_file() {
        eprintln!("skipping: model not found at {model_path}");
        return;
    }
    let config = YoloConfig {
        model_path,
        confidence_threshold: 0.5,
        interval_ms: 1_000,
        detect_person: true,
        detect_car: true,
        device: InferenceDevice::Cpu,
        onnxruntime_path: None,
    };
    let mut engine = YoloEngine::load(&config).expect("load YOLO model");
    let (width, height) = engine.input_size();
    assert!(width >= 320 && height >= 320);

    // A neutral image should run end-to-end without crashes. Depending on the
    // model it may legitimately contain zero detections.
    let image = RgbImage::from_pixel(width, height, Rgb([114, 114, 114]));
    let filter = ClassFilter {
        confidence_threshold: config.confidence_threshold,
        detect_person: true,
        detect_car: true,
    };
    let detections = engine.infer(&image, filter).expect("run inference");
    for detection in detections {
        assert!(matches!(
            detection.class,
            yolocam::detection::DetectedClass::Person | yolocam::detection::DetectedClass::Car
        ));
        assert!(detection.confidence >= config.confidence_threshold);
        assert!(detection.bbox.x >= 0.0 && detection.bbox.y >= 0.0);
        assert!(detection.bbox.x + detection.bbox.w <= 1.0001);
        assert!(detection.bbox.y + detection.bbox.h <= 1.0001);
    }
}
