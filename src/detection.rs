use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use font8x8::{UnicodeFonts, BASIC_FONTS};
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, Rgb, RgbImage};
use serde::Serialize;
use thiserror::Error;

pub const COCO_PERSON: usize = 0;
pub const COCO_CAR: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DetectedClass {
    Person,
    Car,
}

impl DetectedClass {
    pub fn coco_id(self) -> usize {
        match self {
            Self::Person => COCO_PERSON,
            Self::Car => COCO_CAR,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Car => "car",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Person => "Person",
            Self::Car => "Car",
        }
    }

    pub fn color(self) -> Rgb<u8> {
        match self {
            Self::Person => Rgb([34, 197, 94]), // green
            Self::Car => Rgb([59, 130, 246]),   // blue
        }
    }

    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "person" => Some(Self::Person),
            "car" => Some(Self::Car),
            _ => None,
        }
    }

    pub fn from_coco_id(id: usize) -> Option<Self> {
        match id {
            COCO_PERSON => Some(Self::Person),
            COCO_CAR => Some(Self::Car),
            _ => None,
        }
    }
}

impl fmt::Display for DetectedClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BBox {
    /// Normalized coordinates in the range 0.0..=1.0.
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl BBox {
    pub fn clamped(mut self) -> Self {
        let x1 = self.x.clamp(0.0, 1.0);
        let y1 = self.y.clamp(0.0, 1.0);
        let x2 = (self.x + self.w).clamp(0.0, 1.0);
        let y2 = (self.y + self.h).clamp(0.0, 1.0);
        self.x = x1;
        self.y = y1;
        self.w = (x2 - x1).max(0.0);
        self.h = (y2 - y1).max(0.0);
        self
    }

    pub fn area(&self) -> f32 {
        self.w.max(0.0) * self.h.max(0.0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Detection {
    pub class: DetectedClass,
    pub confidence: f32,
    pub bbox: BBox,
}

impl Detection {
    pub fn label(&self) -> String {
        format!("{} {:.0}%", self.class.label(), self.confidence * 100.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ClassFilter {
    pub confidence_threshold: f32,
    pub detect_person: bool,
    pub detect_car: bool,
}

impl ClassFilter {
    pub fn accepts(&self, class: DetectedClass, confidence: f32) -> bool {
        if !confidence.is_finite() || confidence < self.confidence_threshold {
            return false;
        }
        match class {
            DetectedClass::Person => self.detect_person,
            DetectedClass::Car => self.detect_car,
        }
    }
}

pub fn count_by_class(detections: &[Detection]) -> (usize, usize) {
    let mut people = 0;
    let mut cars = 0;
    for detection in detections {
        match detection.class {
            DetectedClass::Person => people += 1,
            DetectedClass::Car => cars += 1,
        }
    }
    (people, cars)
}

pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let ax2 = a.x + a.w;
    let ay2 = a.y + a.h;
    let bx2 = b.x + b.w;
    let by2 = b.y + b.h;
    let intersect_w = (ax2.min(bx2) - a.x.max(b.x)).max(0.0);
    let intersect_h = (ay2.min(by2) - a.y.max(b.y)).max(0.0);
    let intersection = intersect_w * intersect_h;
    let union = a.area() + b.area() - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}

#[derive(Debug, Error)]
pub enum ScreenshotError {
    #[error("cannot create screenshot directory: {0}")]
    Directory(#[source] std::io::Error),
    #[error("cannot write screenshot: {0}")]
    Io(#[from] std::io::Error),
    #[error("cannot encode screenshot: {0}")]
    Encode(String),
}

/// Annotate an image with boxes, confidence labels, camera name and timestamp.
pub fn render_annotated(
    image: &RgbImage,
    detections: &[Detection],
    camera_name: &str,
    captured_at: DateTime<Utc>,
) -> RgbImage {
    let mut canvas = image.clone();
    let scale = if canvas.width() >= 960 { 2 } else { 1 };
    let thickness = (canvas.width() / 420).clamp(2, 5);

    for detection in detections {
        let color = detection.class.color();
        let rect = pixel_rect(&canvas, detection.bbox);
        draw_rect(&mut canvas, rect, color, thickness);

        let label = detection.label();
        let text_scale = scale;
        let glyph_size = 8 * text_scale;
        let padding = 3;
        let text_width = label.chars().count() as u32 * glyph_size + padding * 2;
        let text_height = glyph_size + padding * 2;
        let label_x = rect
            .0
            .max(0)
            .min(canvas.width().saturating_sub(text_width) as i32);
        let label_y = if rect.1 >= text_height as i32 {
            rect.1 - text_height as i32
        } else {
            rect.1
        };
        fill_rect(
            &mut canvas,
            (label_x, label_y, text_width, text_height),
            color,
        );
        draw_text(
            &mut canvas,
            label_x + padding as i32,
            label_y + padding as i32,
            &label,
            Rgb([255, 255, 255]),
            text_scale,
        );
    }

    let header = format!(
        "{}  {}",
        sanitize_camera_name(camera_name),
        captured_at.format("%Y-%m-%d %H:%M:%S UTC")
    );
    let header_scale = scale;
    let glyph_size = 8 * header_scale;
    let padding = 4;
    let header_height = glyph_size + padding * 2;
    let canvas_width = canvas.width();
    fill_rect(
        &mut canvas,
        (0, 0, canvas_width, header_height),
        Rgb([8, 12, 20]),
    );
    draw_text(
        &mut canvas,
        padding as i32,
        padding as i32,
        &header,
        Rgb([226, 232, 240]),
        header_scale,
    );

    canvas
}

/// Save one annotated JPEG for an analyzed frame. Callers save at most one
/// screenshot per frame, regardless of how many boxes are present.
pub fn save_screenshot(
    image: &RgbImage,
    detections: &[Detection],
    camera_name: &str,
    captured_at: DateTime<Utc>,
    directory: &Path,
) -> Result<String, ScreenshotError> {
    fs::create_dir_all(directory).map_err(ScreenshotError::Directory)?;
    let annotated = render_annotated(image, detections, camera_name, captured_at);
    let safe_name = sanitize_camera_name(camera_name);

    let mut stamp = captured_at;
    let mut path = PathBuf::new();
    let mut filename = String::new();
    for _ in 0..1_000 {
        filename = format!(
            "{}_{}.jpg",
            safe_name,
            stamp.format("%Y-%m-%d_%H-%M-%S-%3f")
        );
        path = directory.join(&filename);
        if !path.exists() {
            break;
        }
        stamp += ChronoDuration::milliseconds(1);
    }
    if path.exists() {
        return Err(ScreenshotError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique screenshot filename",
        )));
    }

    let file = File::create(&path)?;
    let mut writer = BufWriter::new(file);
    {
        let mut encoder = JpegEncoder::new_with_quality(&mut writer, 88);
        encoder
            .encode(
                annotated.as_raw(),
                annotated.width(),
                annotated.height(),
                ExtendedColorType::Rgb8,
            )
            .map_err(|error| ScreenshotError::Encode(error.to_string()))?;
    }
    writer.flush()?;
    Ok(filename)
}

pub fn sanitize_camera_name(name: &str) -> String {
    let mut sanitized = String::with_capacity(name.len());
    let mut last_dash = false;
    for character in name.trim().chars() {
        let valid = character.is_ascii_alphanumeric();
        if valid {
            sanitized.push(character.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            sanitized.push('-');
            last_dash = true;
        }
        if sanitized.len() >= 40 {
            break;
        }
    }
    let trimmed = sanitized.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "camera".to_string()
    } else {
        trimmed
    }
}

/// Delete only files matching the app's generated screenshot naming scheme
/// and older than the retention window. Unrelated files are left alone.
pub fn cleanup_screenshots(
    directory: &Path,
    retention_days: u32,
) -> Result<usize, ScreenshotError> {
    if !directory.exists() {
        return Ok(0);
    }
    let retention = Duration::from_secs(u64::from(retention_days.max(1)) * 86_400);
    let cutoff = SystemTime::now()
        .checked_sub(retention)
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut removed = 0usize;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !is_managed_screenshot(name) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if modified < cutoff && fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

pub fn is_managed_screenshot(filename: &str) -> bool {
    let Some(stem) = filename.strip_suffix(".jpg") else {
        return false;
    };
    let mut parts = stem.rsplit('_');
    let Some(time_part) = parts.next() else {
        return false;
    };
    let Some(date_part) = parts.next() else {
        return false;
    };
    let Some(prefix) = parts.next() else {
        return false;
    };
    if prefix.is_empty() {
        return false;
    }
    is_date_part(date_part) && is_time_part(time_part)
}

fn is_date_part(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let Some(year) = parse_digits(&value[0..4]) else {
        return false;
    };
    let Some(month) = parse_digits(&value[5..7]) else {
        return false;
    };
    let Some(day) = parse_digits(&value[8..10]) else {
        return false;
    };
    (1..=9999).contains(&year) && (1..=12).contains(&month) && (1..=31).contains(&day)
}

fn is_time_part(value: &str) -> bool {
    let bytes = value.as_bytes();
    // HH-MM-SS-mmm
    if bytes.len() != 12 || bytes[2] != b'-' || bytes[5] != b'-' || bytes[8] != b'-' {
        return false;
    }
    let Some(hour) = parse_digits(&value[0..2]) else {
        return false;
    };
    let Some(minute) = parse_digits(&value[3..5]) else {
        return false;
    };
    let Some(second) = parse_digits(&value[6..8]) else {
        return false;
    };
    let Some(millis) = parse_digits(&value[9..12]) else {
        return false;
    };
    hour <= 23 && minute <= 59 && second <= 59 && millis <= 999
}

fn parse_digits(value: &str) -> Option<u32> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

fn pixel_rect(image: &RgbImage, bbox: BBox) -> (i32, i32, u32, u32) {
    let x1 = (bbox.x * image.width() as f32).floor().max(0.0) as i32;
    let y1 = (bbox.y * image.height() as f32).floor().max(0.0) as i32;
    let x2 = ((bbox.x + bbox.w) * image.width() as f32)
        .ceil()
        .min(image.width() as f32) as i32;
    let y2 = ((bbox.y + bbox.h) * image.height() as f32)
        .ceil()
        .min(image.height() as f32) as i32;
    (
        x1.min(image.width() as i32 - 1),
        y1.min(image.height() as i32 - 1),
        (x2 - x1).max(1) as u32,
        (y2 - y1).max(1) as u32,
    )
}

fn draw_rect(
    image: &mut RgbImage,
    (x, y, width, height): (i32, i32, u32, u32),
    color: Rgb<u8>,
    thickness: u32,
) {
    if width == 0 || height == 0 {
        return;
    }
    let image_width = image.width() as i32;
    let image_height = image.height() as i32;
    for offset in 0..thickness {
        let top = y + offset as i32;
        let bottom = y + height as i32 - 1 - offset as i32;
        let left = x + offset as i32;
        let right = x + width as i32 - 1 - offset as i32;
        if top >= 0 && top < image_height {
            for px in x.max(0)..(x + width as i32).min(image_width) {
                image.put_pixel(px as u32, top as u32, color);
            }
        }
        if bottom >= 0 && bottom < image_height {
            for px in x.max(0)..(x + width as i32).min(image_width) {
                image.put_pixel(px as u32, bottom as u32, color);
            }
        }
        if left >= 0 && left < image_width {
            for py in y.max(0)..(y + height as i32).min(image_height) {
                image.put_pixel(left as u32, py as u32, color);
            }
        }
        if right >= 0 && right < image_width {
            for py in y.max(0)..(y + height as i32).min(image_height) {
                image.put_pixel(right as u32, py as u32, color);
            }
        }
    }
}

fn fill_rect(image: &mut RgbImage, (x, y, width, height): (i32, i32, u32, u32), color: Rgb<u8>) {
    let x_start = x.max(0) as u32;
    let y_start = y.max(0) as u32;
    let x_end = (x + width as i32).min(image.width() as i32).max(0) as u32;
    let y_end = (y + height as i32).min(image.height() as i32).max(0) as u32;
    for py in y_start..y_end {
        for px in x_start..x_end {
            image.put_pixel(px, py, color);
        }
    }
}

fn draw_text(image: &mut RgbImage, x: i32, y: i32, text: &str, color: Rgb<u8>, scale: u32) {
    let mut cursor_x = x;
    for character in text.chars() {
        let Some(glyph) = BASIC_FONTS.get(character) else {
            cursor_x += 8 * scale as i32;
            continue;
        };
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..8u32 {
                if *bits & (1 << col) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cursor_x + (col * scale + sx) as i32;
                            let py = y + (row as u32 * scale + sy) as i32;
                            if px >= 0
                                && py >= 0
                                && (px as u32) < image.width()
                                && (py as u32) < image.height()
                            {
                                image.put_pixel(px as u32, py as u32, color);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 8 * scale as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_detection() -> Detection {
        Detection {
            class: DetectedClass::Person,
            confidence: 0.91,
            bbox: BBox {
                x: 0.1,
                y: 0.2,
                w: 0.3,
                h: 0.4,
            },
        }
    }

    #[test]
    fn filter_accepts_only_enabled_classes_above_threshold() {
        let filter = ClassFilter {
            confidence_threshold: 0.5,
            detect_person: true,
            detect_car: false,
        };
        assert!(filter.accepts(DetectedClass::Person, 0.5));
        assert!(filter.accepts(DetectedClass::Person, 0.9));
        assert!(!filter.accepts(DetectedClass::Person, 0.49));
        assert!(!filter.accepts(DetectedClass::Car, 0.99));
        assert!(!filter.accepts(DetectedClass::Person, f32::NAN));
    }

    #[test]
    fn counts_only_people_and_cars() {
        let detections = vec![
            sample_detection(),
            Detection {
                class: DetectedClass::Car,
                confidence: 0.8,
                bbox: BBox {
                    x: 0.5,
                    y: 0.5,
                    w: 0.1,
                    h: 0.1,
                },
            },
            sample_detection(),
        ];
        assert_eq!(count_by_class(&detections), (2, 1));
    }

    #[test]
    fn iou_handles_overlap_and_disjoint_boxes() {
        let a = BBox {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        let b = BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        assert!((iou(&a, &b) - 1.0 / 7.0).abs() < 1e-5);
        let c = BBox {
            x: 0.8,
            y: 0.8,
            w: 0.1,
            h: 0.1,
        };
        assert_eq!(iou(&a, &c), 0.0);
    }

    #[test]
    fn camera_name_is_safe_for_filenames() {
        assert_eq!(sanitize_camera_name("Front Camera"), "front-camera");
        assert_eq!(sanitize_camera_name("../../etc/passwd"), "etc-passwd");
        assert_eq!(sanitize_camera_name("   "), "camera");
        assert_eq!(sanitize_camera_name("Cam_01"), "cam-01");
    }

    #[test]
    fn screenshot_filename_pattern_is_recognized() {
        assert!(is_managed_screenshot(
            "front-camera_2026-11-23_19-30-01-123.jpg"
        ));
        assert!(!is_managed_screenshot("family_photo.jpg"));
        assert!(!is_managed_screenshot(
            "front-camera_2026-11-23_19-30-01-123.png"
        ));
        assert!(!is_managed_screenshot("evil_2026-99-99_99-99-99-999.jpg"));
    }

    #[test]
    fn rendering_and_saving_screenshot_works() {
        let mut image = RgbImage::from_pixel(320, 240, Rgb([20, 20, 20]));
        image.put_pixel(10, 10, Rgb([1, 2, 3]));
        let annotated = render_annotated(
            &image,
            &[sample_detection()],
            "Front Camera",
            DateTime::UNIX_EPOCH,
        );
        assert_eq!(annotated.dimensions(), (320, 240));
        assert_ne!(annotated.get_pixel(10, 10), image.get_pixel(10, 10));

        let dir = tempdir().unwrap();
        let filename = save_screenshot(
            &image,
            &[sample_detection()],
            "Front Camera",
            DateTime::UNIX_EPOCH,
            dir.path(),
        )
        .unwrap();
        assert!(is_managed_screenshot(&filename), "{filename}");
        assert!(dir.path().join(&filename).is_file());
        let decoded = image::open(dir.path().join(&filename)).unwrap();
        assert_eq!(decoded.width(), 320);
    }

    #[test]
    fn cleanup_removes_only_expired_managed_screenshots() {
        let dir = tempdir().unwrap();
        let managed = dir.path().join("cam_2020-01-01_00-00-00-000.jpg");
        let unrelated = dir.path().join("important.jpg");
        fs::write(&managed, b"x").unwrap();
        fs::write(&unrelated, b"x").unwrap();
        let old = SystemTime::now() - Duration::from_secs(86_400 * 30);
        let times = std::fs::FileTimes::new().set_modified(old);
        File::options()
            .write(true)
            .open(&managed)
            .unwrap()
            .set_times(times)
            .unwrap();
        let removed = cleanup_screenshots(dir.path(), 7).unwrap();
        assert_eq!(removed, 1);
        assert!(!managed.exists());
        assert!(unrelated.exists());
    }
}
