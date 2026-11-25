use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use image::imageops::FilterType;
use image::{ImageFormat, Rgb, RgbImage};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, TensorElementType, ValueType};
use thiserror::Error;

use crate::config::{InferenceDevice, YoloConfig};
use crate::detection::{iou, BBox, ClassFilter, DetectedClass, Detection};

pub const NMS_IOU_THRESHOLD: f32 = 0.45;
const LETTERBOX_PADDING: u8 = 114;

/// ONNX Runtime permits only one environment per process, so every model
/// (including benchmark candidates) shares this one.
static SHARED_ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
static ENVIRONMENT_INIT: Mutex<()> = Mutex::new(());

#[derive(Debug, Error)]
pub enum YoloError {
    #[error("ONNX Runtime library not found. Place onnxruntime.dll / libonnxruntime.so next to the executable, or set yolo.onnxruntime_path / ORT_DYLIB_PATH. Tried: {0}")]
    RuntimeNotFound(String),
    #[error("failed to load ONNX Runtime from {path}: {message}")]
    RuntimeLoad { path: String, message: String },
    #[error("YOLO model file not found: {0}")]
    ModelNotFound(String),
    #[error("invalid YOLO model: {0}")]
    InvalidModel(String),
    #[error("unsupported YOLO output format: {0}")]
    UnsupportedOutput(String),
    #[error("YOLO inference failed: {0}")]
    Inference(String),
    #[error("image decoding failed: {0}")]
    Image(String),
    #[error("execution provider is not available in this build: {0}")]
    UnsupportedDevice(String),
}

pub struct YoloEngine {
    _environment: Environment,
    session: Session,
    input_name: String,
    output_name: String,
    input_width: u32,
    input_height: u32,
}

impl YoloEngine {
    pub fn load(config: &YoloConfig) -> Result<Self, YoloError> {
        let model_path = Path::new(&config.model_path);
        if !model_path.is_file() {
            return Err(YoloError::ModelNotFound(config.model_path.clone()));
        }

        let runtime_path = resolve_onnxruntime_path(config)?;
        let environment = shared_environment(&runtime_path, config)?;

        let threads = std::thread::available_parallelism()
            .map(|count| (count.get() / 2).clamp(1, 4))
            .unwrap_or(2);
        let session = Session::builder(&environment)
            .map_err(|error| YoloError::Inference(error.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| YoloError::Inference(error.to_string()))?
            .with_intra_threads(threads)
            .map_err(|error| YoloError::Inference(error.to_string()))?
            .commit_from_file(model_path)
            .map_err(|error| YoloError::InvalidModel(error.to_string()))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| YoloError::InvalidModel("model has no inputs".into()))?;
        let input_name = input.name().to_string();
        let (input_width, input_height) = match input.dtype() {
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                shape,
                ..
            } => {
                return Err(YoloError::InvalidModel(format!(
                    "model input {} uses float16; export a float32 ONNX model (Ultralytics: half=False, dynamic=False)",
                    shape
                )));
            }
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape,
                ..
            } => {
                if shape.len() != 4 {
                    return Err(YoloError::InvalidModel(format!(
                        "expected a rank-4 float32 input, found shape {shape}"
                    )));
                }
                if shape[0] > 0 && shape[0] != 1 {
                    return Err(YoloError::InvalidModel(format!(
                        "expected batch size 1, found input shape {shape}"
                    )));
                }
                if shape[1] != 3 {
                    return Err(YoloError::InvalidModel(format!(
                        "expected 3 input channels (RGB), found input shape {shape}"
                    )));
                }
                let height = if shape[2] > 0 { shape[2] as u32 } else { 640 };
                let width = if shape[3] > 0 { shape[3] as u32 } else { 640 };
                if width < 32 || height < 32 || width > 4096 || height > 4096 {
                    return Err(YoloError::InvalidModel(format!(
                        "unsupported model input size {width}x{height}"
                    )));
                }
                (width, height)
            }
            other => {
                return Err(YoloError::InvalidModel(format!(
                    "expected a float32 NCHW input tensor, found {other:?}"
                )));
            }
        };
        let output_name = session
            .outputs()
            .first()
            .map(|output| output.name().to_string())
            .ok_or_else(|| YoloError::InvalidModel("model has no outputs".into()))?;

        Ok(Self {
            _environment: environment,
            session,
            input_name,
            output_name,
            input_width,
            input_height,
        })
    }

    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn infer(
        &mut self,
        image: &RgbImage,
        filter: ClassFilter,
    ) -> Result<Vec<Detection>, YoloError> {
        let (tensor_data, letterbox) = preprocess(image, self.input_width, self.input_height);
        let tensor = Tensor::from_array((
            vec![1_i64, 3, self.input_height as i64, self.input_width as i64],
            tensor_data,
        ))
        .map_err(|error| YoloError::Inference(error.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|error| YoloError::Inference(error.to_string()))?;
        let output = &outputs[0];
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| YoloError::Inference(error.to_string()))?;
        decode_yolo_output(values, &shape[..], &letterbox, filter)
    }
}

fn shared_environment(runtime_path: &Path, config: &YoloConfig) -> Result<Environment, YoloError> {
    let _guard = ENVIRONMENT_INIT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(environment) = SHARED_ENVIRONMENT.get() {
        return Ok(environment.clone());
    }

    if config.device == InferenceDevice::Cuda && !cfg!(feature = "cuda") {
        return Err(YoloError::UnsupportedDevice(
            "device \"cuda\" requires building with --features cuda".into(),
        ));
    }
    if config.device == InferenceDevice::DirectMl && (!cfg!(feature = "directml") || !cfg!(windows))
    {
        return Err(YoloError::UnsupportedDevice(
            "device \"directml\" requires Windows and a build with --features directml".into(),
        ));
    }

    let mut environment_builder =
        ort::init_from(runtime_path).map_err(|error| YoloError::RuntimeLoad {
            path: runtime_path.display().to_string(),
            message: error.to_string(),
        })?;
    environment_builder = environment_builder.with_telemetry(false);

    #[cfg(feature = "cuda")]
    if config.device == InferenceDevice::Cuda {
        environment_builder = environment_builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()]);
    }
    #[cfg(all(feature = "directml", target_os = "windows"))]
    if config.device == InferenceDevice::DirectMl {
        environment_builder = environment_builder
            .with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()]);
    }

    let environment = environment_builder
        .build()
        .map_err(|error| YoloError::RuntimeLoad {
            path: runtime_path.display().to_string(),
            message: error.to_string(),
        })?;
    let _ = SHARED_ENVIRONMENT.set(environment.clone());
    Ok(environment)
}

fn resolve_onnxruntime_path(config: &YoloConfig) -> Result<PathBuf, YoloError> {
    let mut tried: Vec<String> = Vec::new();

    if let Some(configured) = config
        .onnxruntime_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let path = PathBuf::from(configured);
        if path.is_file() {
            return Ok(path);
        }
        tried.push(configured.to_string());
    }

    if let Ok(value) = std::env::var("ORT_DYLIB_PATH") {
        let value = value.trim();
        if !value.is_empty() {
            let path = PathBuf::from(value);
            if path.is_file() {
                return Ok(path);
            }
            tried.push(value.to_string());
        }
    }

    let library_names = runtime_library_names();
    let mut search_dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            search_dirs.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        search_dirs.push(cwd);
    }
    for dir in &search_dirs {
        for name in &library_names {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
            tried.push(candidate.display().to_string());
        }
    }

    // Let the OS loader search PATH / LD_LIBRARY_PATH as a last resort.
    if let Some(name) = library_names.first() {
        return Ok(PathBuf::from(name));
    }
    Err(YoloError::RuntimeNotFound(tried.join(", ")))
}

fn runtime_library_names() -> Vec<&'static str> {
    #[cfg(target_os = "windows")]
    {
        vec!["onnxruntime.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        vec![
            "libonnxruntime.dylib",
            "libonnxruntime.1.31.1.dylib",
            "libonnxruntime.1.31.0.dylib",
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        vec![
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "libonnxruntime.so.1.31.1",
            "libonnxruntime.so.1.31.0",
        ]
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        vec!["onnxruntime.dll"]
    }
}

pub fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage, YoloError> {
    let image = image::load_from_memory_with_format(bytes, ImageFormat::Jpeg)
        .map_err(|error| YoloError::Image(error.to_string()))?;
    let image = image.to_rgb8();
    if image.width() == 0 || image.height() == 0 {
        return Err(YoloError::Image("decoded image has zero dimensions".into()));
    }
    if u64::from(image.width()) * u64::from(image.height()) > 32_000_000 {
        return Err(YoloError::Image(
            "decoded frame is larger than 32 megapixels".into(),
        ));
    }
    Ok(image)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale_x: f32,
    pub scale_y: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub resized_w: u32,
    pub resized_h: u32,
    pub source_w: u32,
    pub source_h: u32,
    pub target_w: u32,
    pub target_h: u32,
}

impl Letterbox {
    /// Map raw YOLO cx/cy/w/h in model-input pixels back to a normalized
    /// bounding box in the original image.
    pub fn map_box(&self, cx: f32, cy: f32, width: f32, height: f32) -> Option<BBox> {
        if !cx.is_finite()
            || !cy.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || width <= 0.0
            || height <= 0.0
        {
            return None;
        }
        let x1 = (cx - width / 2.0 - self.pad_x) / self.scale_x;
        let y1 = (cy - height / 2.0 - self.pad_y) / self.scale_y;
        let x2 = (cx + width / 2.0 - self.pad_x) / self.scale_x;
        let y2 = (cy + height / 2.0 - self.pad_y) / self.scale_y;
        let bbox = BBox {
            x: x1 / self.source_w as f32,
            y: y1 / self.source_h as f32,
            w: (x2 - x1) / self.source_w as f32,
            h: (y2 - y1) / self.source_h as f32,
        }
        .clamped();
        if bbox.w <= 0.0 || bbox.h <= 0.0 {
            None
        } else {
            Some(bbox)
        }
    }
}

/// Letterbox resize and convert RGB to a normalized NCHW f32 tensor.
pub fn preprocess(image: &RgbImage, target_w: u32, target_h: u32) -> (Vec<f32>, Letterbox) {
    assert!(target_w > 0 && target_h > 0);
    let source_w = image.width();
    let source_h = image.height();
    let scale = (target_w as f32 / source_w as f32).min(target_h as f32 / source_h as f32);
    let resized_w = ((source_w as f32 * scale).round() as u32).clamp(1, target_w);
    let resized_h = ((source_h as f32 * scale).round() as u32).clamp(1, target_h);
    let pad_x = (target_w - resized_w) / 2;
    let pad_y = (target_h - resized_h) / 2;
    let resized = image::imageops::resize(image, resized_w, resized_h, FilterType::Triangle);

    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([LETTERBOX_PADDING; 3]));
    image::imageops::replace(&mut canvas, &resized, i64::from(pad_x), i64::from(pad_y));

    let plane = (target_w * target_h) as usize;
    let mut data = vec![f32::from(LETTERBOX_PADDING) / 255.0; plane * 3];
    for (x, y, pixel) in canvas.enumerate_pixels() {
        let index = y as usize * target_w as usize + x as usize;
        data[index] = f32::from(pixel[0]) / 255.0;
        data[plane + index] = f32::from(pixel[1]) / 255.0;
        data[plane * 2 + index] = f32::from(pixel[2]) / 255.0;
    }

    let letterbox = Letterbox {
        scale_x: resized_w as f32 / source_w as f32,
        scale_y: resized_h as f32 / source_h as f32,
        pad_x: pad_x as f32,
        pad_y: pad_y as f32,
        resized_w,
        resized_h,
        source_w,
        source_h,
        target_w,
        target_h,
    };
    (data, letterbox)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputLayout {
    ChannelMajor { channels: usize, anchors: usize },
    AnchorMajor { channels: usize, anchors: usize },
}

impl OutputLayout {
    fn channels(self) -> usize {
        match self {
            Self::ChannelMajor { channels, .. } | Self::AnchorMajor { channels, .. } => channels,
        }
    }

    fn anchors(self) -> usize {
        match self {
            Self::ChannelMajor { anchors, .. } | Self::AnchorMajor { anchors, .. } => anchors,
        }
    }

    fn get(self, data: &[f32], anchor: usize, channel: usize) -> f32 {
        match self {
            Self::ChannelMajor { .. } => data[channel * self.anchors() + anchor],
            Self::AnchorMajor { .. } => data[anchor * self.channels() + channel],
        }
    }
}

fn parse_layout(shape: &[i64]) -> Result<OutputLayout, YoloError> {
    if shape.len() != 3 {
        return Err(YoloError::UnsupportedOutput(format!(
            "expected a rank-3 tensor [1, channels, anchors] or [1, anchors, channels], found {shape:?}"
        )));
    }
    if shape[0] != 1 {
        return Err(YoloError::UnsupportedOutput(format!(
            "expected batch size 1, found {shape:?}"
        )));
    }
    let a = usize::try_from(shape[1]).ok();
    let b = usize::try_from(shape[2]).ok();
    let supports = |channels: usize| channels == 84 || channels == 85;
    if let Some(channels) = a.filter(|channels| supports(*channels)) {
        let anchors = b.filter(|anchors| *anchors > channels).ok_or_else(|| {
            YoloError::UnsupportedOutput(format!("invalid anchor count in shape {shape:?}"))
        })?;
        return Ok(OutputLayout::ChannelMajor { channels, anchors });
    }
    if let Some(channels) = b.filter(|channels| supports(*channels)) {
        let anchors = a.filter(|anchors| *anchors > channels).ok_or_else(|| {
            YoloError::UnsupportedOutput(format!("invalid anchor count in shape {shape:?}"))
        })?;
        return Ok(OutputLayout::AnchorMajor { channels, anchors });
    }
    Err(YoloError::UnsupportedOutput(format!(
        "expected COCO YOLOv8/YOLO11 output with 84 channels (4 box + 80 classes) or legacy 85 channels (with objectness), found {shape:?}. Export with: yolo export model=yolo11n.pt format=onnx imgsz=640"
    )))
}

pub fn decode_yolo_output(
    data: &[f32],
    shape: &[i64],
    letterbox: &Letterbox,
    filter: ClassFilter,
) -> Result<Vec<Detection>, YoloError> {
    let layout = parse_layout(shape)?;
    let channels = layout.channels();
    let anchors = layout.anchors();
    let expected = channels
        .checked_mul(anchors)
        .ok_or_else(|| YoloError::UnsupportedOutput("output tensor is too large".into()))?;
    if data.len() < expected {
        return Err(YoloError::UnsupportedOutput(format!(
            "output has {} values, expected {expected}",
            data.len()
        )));
    }
    let legacy_objectness = channels == 85;
    let class_offset = if legacy_objectness { 5 } else { 4 };

    let classes: [DetectedClass; 2] = [DetectedClass::Person, DetectedClass::Car];
    let mut detections = Vec::with_capacity(16);
    for anchor in 0..anchors {
        let objectness = if legacy_objectness {
            layout.get(data, anchor, 4)
        } else {
            1.0
        };
        if !objectness.is_finite() || objectness <= 0.0 {
            continue;
        }
        for class in classes {
            if !class_enabled(filter, class) {
                continue;
            }
            let class_index = class_offset + class.coco_id();
            let score = layout.get(data, anchor, class_index) * objectness;
            if !filter.accepts(class, score) {
                continue;
            }
            let cx = layout.get(data, anchor, 0);
            let cy = layout.get(data, anchor, 1);
            let width = layout.get(data, anchor, 2);
            let height = layout.get(data, anchor, 3);
            if let Some(bbox) = letterbox.map_box(cx, cy, width, height) {
                detections.push(Detection {
                    class,
                    confidence: score.clamp(0.0, 1.0),
                    bbox,
                });
            }
        }
    }
    Ok(non_max_suppression(detections, NMS_IOU_THRESHOLD))
}

fn class_enabled(filter: ClassFilter, class: DetectedClass) -> bool {
    match class {
        DetectedClass::Person => filter.detect_person,
        DetectedClass::Car => filter.detect_car,
    }
}

pub fn non_max_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for detection in detections {
        let overlaps = kept.iter().any(|existing| {
            existing.class == detection.class
                && iou(&existing.bbox, &detection.bbox) > iou_threshold
        });
        if !overlaps {
            kept.push(detection);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_letterbox() -> Letterbox {
        Letterbox {
            scale_x: 1.0,
            scale_y: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
            resized_w: 640,
            resized_h: 640,
            source_w: 640,
            source_h: 640,
            target_w: 640,
            target_h: 640,
        }
    }

    fn make_channel_major(entries: &[(usize, f32, f32, f32, f32, f32, f32)]) -> Vec<f32> {
        // entries: anchor, cx, cy, w, h, person_score, car_score
        let channels = 84usize;
        let anchors = 8400usize;
        let mut data = vec![0f32; channels * anchors];
        for (anchor, cx, cy, w, h, person, car) in entries {
            data[*anchor] = *cx;
            data[anchors + *anchor] = *cy;
            data[2 * anchors + *anchor] = *w;
            data[3 * anchors + *anchor] = *h;
            data[(4 + COCO_PERSON_INDEX) * anchors + *anchor] = *person;
            data[(4 + COCO_CAR_INDEX) * anchors + *anchor] = *car;
        }
        data
    }

    const COCO_PERSON_INDEX: usize = 0;
    const COCO_CAR_INDEX: usize = 2;

    #[test]
    fn letterbox_geometry_for_wide_and_tall_images() {
        let wide = RgbImage::new(1280, 720);
        let (data, lb) = preprocess(&wide, 640, 640);
        assert_eq!(lb.resized_w, 640);
        assert_eq!(lb.resized_h, 360);
        assert_eq!(lb.pad_y, 140.0);
        assert_eq!(lb.pad_x, 0.0);
        assert_eq!(data.len(), 3 * 640 * 640);
        // Top padding is the letterbox gray.
        assert!((data[0] - 114.0 / 255.0).abs() < 1e-6);

        let tall = RgbImage::new(480, 960);
        let (_, lb) = preprocess(&tall, 640, 640);
        assert_eq!(lb.resized_w, 320);
        assert_eq!(lb.resized_h, 640);
        assert_eq!(lb.pad_x, 160.0);
        assert_eq!(lb.pad_y, 0.0);
    }

    #[test]
    fn box_maps_back_through_letterbox() {
        let wide = RgbImage::new(1280, 720);
        let (_, lb) = preprocess(&wide, 640, 640);
        // A centered 64x48 model-space box around the middle.
        let bbox = lb.map_box(320.0, 320.0, 64.0, 48.0).unwrap();
        // scale=0.5, pad_y=140: x1=(288)/0.5=576, y1=(296-140)/0.5=312.
        assert!((bbox.x - 576.0 / 1280.0).abs() < 1e-5);
        assert!((bbox.y - 312.0 / 720.0).abs() < 1e-5);
        assert!((bbox.w - 128.0 / 1280.0).abs() < 1e-5);
        assert!((bbox.h - 96.0 / 720.0).abs() < 1e-5);
    }

    #[test]
    fn decodes_channel_major_yolo11_output() {
        let data = make_channel_major(&[(100, 320.0, 320.0, 100.0, 200.0, 0.91, 0.01)]);
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: true,
        };
        let detections =
            decode_yolo_output(&data, &[1, 84, 8400], &identity_letterbox(), filter).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class, DetectedClass::Person);
        assert!((detections[0].confidence - 0.91).abs() < 1e-6);
        assert!((detections[0].bbox.x - 0.421875).abs() < 1e-5);
        assert!((detections[0].bbox.w - 0.15625).abs() < 1e-5);
    }

    #[test]
    fn decodes_anchor_major_output_and_filters_classes() {
        let anchors = 8400usize;
        let channels = 84usize;
        let mut data = vec![0f32; channels * anchors];
        let row = 7 * channels;
        data[row] = 100.0;
        data[row + 1] = 100.0;
        data[row + 2] = 50.0;
        data[row + 3] = 50.0;
        data[row + 4 + 2] = 0.8;
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: true,
        };
        let detections =
            decode_yolo_output(&data, &[1, 8400, 84], &identity_letterbox(), filter).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class, DetectedClass::Car);
        assert!((detections[0].confidence - 0.8).abs() < 1e-6);

        let person_only = ClassFilter {
            detect_car: false,
            ..filter
        };
        assert!(
            decode_yolo_output(&data, &[1, 8400, 84], &identity_letterbox(), person_only)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn nms_removes_duplicate_boxes_per_class() {
        let box_a = Detection {
            class: DetectedClass::Person,
            confidence: 0.9,
            bbox: BBox {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
        };
        let box_b = Detection {
            confidence: 0.8,
            ..box_a.clone()
        };
        let car = Detection {
            class: DetectedClass::Car,
            confidence: 0.7,
            bbox: BBox {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
        };
        let kept = non_max_suppression(vec![box_a, box_b, car], 0.45);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].confidence, 0.9);
        assert_eq!(kept[1].class, DetectedClass::Car);
    }

    #[test]
    fn rejects_unknown_output_layout() {
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: true,
        };
        let error = decode_yolo_output(&[0.0; 10], &[1, 10], &identity_letterbox(), filter)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rank-3"), "{error}");
        let error = decode_yolo_output(&[0.0; 160], &[1, 20, 8], &identity_letterbox(), filter)
            .unwrap_err()
            .to_string();
        assert!(error.contains("84 channels"), "{error}");
    }

    #[test]
    fn confidence_threshold_is_applied() {
        let data = make_channel_major(&[
            (1, 100.0, 100.0, 20.0, 20.0, 0.49, 0.0),
            (2, 200.0, 200.0, 20.0, 20.0, 0.51, 0.0),
        ]);
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: true,
        };
        let detections =
            decode_yolo_output(&data, &[1, 84, 8400], &identity_letterbox(), filter).unwrap();
        assert_eq!(detections.len(), 1);
        assert!((detections[0].confidence - 0.51).abs() < 1e-6);
    }

    #[test]
    fn supports_legacy_85_channel_output() {
        let anchors = 100usize;
        let channels = 85usize;
        let mut data = vec![0f32; channels * anchors];
        let anchor = 0usize;
        data[anchor] = 320.0;
        data[anchors + anchor] = 320.0;
        data[2 * anchors + anchor] = 80.0;
        data[3 * anchors + anchor] = 80.0;
        data[4 * anchors + anchor] = 0.9;
        data[(5 + 2) * anchors + anchor] = 0.95;
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: true,
        };
        let detections =
            decode_yolo_output(&data, &[1, 85, 100], &identity_letterbox(), filter).unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class, DetectedClass::Car);
        assert!((detections[0].confidence - 0.855).abs() < 1e-5);
    }

    #[test]
    fn jpeg_decoder_reports_errors_instead_of_panicking() {
        let error = decode_jpeg(b"not a jpeg").unwrap_err().to_string();
        assert!(error.contains("image decoding failed"), "{error}");
    }
}
