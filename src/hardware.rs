use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::{SecondsFormat, Utc};
use image::{Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::{Config, ModelSize, RebenchmarkPolicy, YoloConfig};
use crate::detection::ClassFilter;
use crate::models::{self, ModelCandidate};
use crate::yolo::YoloEngine;

const BENCHMARK_VERSION: u32 = 1;
const WARMUP_RUNS: usize = 3;
const TIMED_RUNS: usize = 12;
const BENCHMARK_BUDGET: Duration = Duration::from_secs(25);
const CACHE_PATH: &str = "models/auto/selection.json";

#[derive(Debug, Error)]
pub enum HardwareError {
    #[error("no usable YOLO model is available for the hardware test: {0}")]
    NoModel(String),
    #[error("model selection failed: {0}")]
    Model(#[from] models::ModelError),
    #[error("hardware benchmark failed: {0}")]
    Benchmark(String),
    #[error("hardware benchmark was interrupted")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: String,
    pub cores: usize,
    pub total_memory_mb: Option<u64>,
    pub os: String,
    pub arch: String,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoReport {
    pub version: u32,
    pub fingerprint: String,
    pub measured_at: String,
    pub hardware: HardwareInfo,
    pub target_ms: f32,
    pub probe_id: String,
    pub probe_p50_ms: f32,
    pub probe_p90_ms: f32,
    pub selected_id: String,
    pub selected_model_path: String,
    pub selected_p50_ms: Option<f32>,
    pub selected_p90_ms: Option<f32>,
    #[serde(default)]
    pub timings: BTreeMap<String, f32>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub cached: bool,
}

#[derive(Debug, Clone, Copy)]
struct Measurement {
    p50_ms: f32,
    p90_ms: f32,
}

pub fn describe(device: crate::config::InferenceDevice) -> HardwareInfo {
    let mut system = sysinfo::System::new_all();
    system.refresh_all();
    let cpu = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_string())
        .filter(|brand| !brand.is_empty())
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let logical = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or_else(|_| system.cpus().len().max(1));
    let physical = system.physical_core_count();
    let total_memory_mb = match system.total_memory() {
        0 => None,
        // sysinfo reports bytes on all supported platforms for this version.
        bytes => Some((bytes / 1_048_576).max(1)),
    };
    HardwareInfo {
        cpu,
        cores: physical.unwrap_or(logical).max(1),
        total_memory_mb,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        device: device.as_str().to_string(),
    }
}

pub fn target_ms(yolo: &YoloConfig) -> f32 {
    if yolo.auto.target_inference_ms > 0 {
        yolo.auto.target_inference_ms as f32
    } else {
        (yolo.interval_ms as f32 * 0.40).clamp(100.0, 1_500.0)
    }
}

fn fingerprint(hardware: &HardwareInfo) -> String {
    format!(
        "v{}|{}|{}|{}|{}|{}|{}",
        BENCHMARK_VERSION,
        hardware.cpu,
        hardware.cores,
        hardware.total_memory_mb.unwrap_or(0),
        hardware.os,
        hardware.arch,
        hardware.device
    )
}

fn effective_max(
    hardware: &HardwareInfo,
    requested: ModelSize,
    notes: &mut Vec<String>,
) -> ModelSize {
    match hardware.total_memory_mb {
        Some(memory) if memory < 1_536 && requested.rank() > ModelSize::N.rank() => {
            notes.push(format!(
                "Only {memory} MB RAM detected; limiting auto-selection to yolo11n."
            ));
            ModelSize::N
        }
        _ => requested,
    }
}

fn low_memory_emergency(hardware: &HardwareInfo) -> bool {
    hardware
        .total_memory_mb
        .map(|memory| memory < 1_024)
        .unwrap_or(false)
}

/// Run (or reuse) the hardware test and return the selected model.
///
/// `force` bypasses the cached result and is used by the dashboard's
/// "Re-run hardware test" action.
pub fn resolve(
    config: &Config,
    force: bool,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(String),
) -> Result<AutoReport, HardwareError> {
    let yolo = &config.yolo;
    let hardware = describe(yolo.device);
    let fingerprint = fingerprint(&hardware);
    let target = target_ms(yolo);
    let mut notes = Vec::new();
    let max_model = effective_max(&hardware, yolo.auto.max_model, &mut notes);

    if !force {
        if let Some(mut cache) = load_cache() {
            let policy_allows_cache = match yolo.auto.rebenchmark {
                RebenchmarkPolicy::Always => false,
                RebenchmarkPolicy::Never => true,
                RebenchmarkPolicy::Hardware => {
                    cache.version == BENCHMARK_VERSION && cache.fingerprint == fingerprint
                }
            };
            if policy_allows_cache {
                if let Some(report) = reuse_cache(&mut cache, &hardware, target, max_model) {
                    progress(format!(
                        "Using cached hardware result: {} (p90 {:.0} ms)",
                        report.selected_id,
                        report.selected_p90_ms.unwrap_or(0.0)
                    ));
                    return Ok(report);
                }
            }
        }
    }

    if stop.load(Ordering::Acquire) {
        return Err(HardwareError::Cancelled);
    }

    let mut timings: BTreeMap<String, f32> = BTreeMap::new();
    let (probe_path, probe_id, probe) =
        measure_probe(yolo, stop, progress, &mut notes).map_err(HardwareError::NoModel)?;

    timings.insert(probe_id.clone(), probe.p90_ms);

    let emergency = low_memory_emergency(&hardware);
    let mut selected: Option<&'static ModelCandidate> = None;
    let mut selected_path: Option<PathBuf> = Some(probe_path.clone());
    let mut selected_measurement: Option<Measurement> = Some(probe);
    let mut selected_id = probe_id.clone();

    if emergency {
        notes.push("Less than 1 GB RAM detected; using the smallest model.".to_string());
        if let Some(candidate) = models::candidate("n320") {
            match ensure_and_measure(candidate, yolo, stop, progress, &mut notes) {
                Ok((path, measurement)) => {
                    timings.insert(candidate.id.to_string(), measurement.p90_ms);
                    selected = Some(candidate);
                    selected_path = Some(path);
                    selected_measurement = Some(measurement);
                    selected_id = candidate.id.to_string();
                }
                Err(error) => notes.push(format!(
                    "Could not switch to {}: {error}; keeping {}.",
                    candidate.id, probe_id
                )),
            }
        }
    } else {
        let candidates: Vec<&'static ModelCandidate> = if yolo.auto.allow_download {
            models::allowed_desc(max_model).collect()
        } else {
            models::allowed_desc(max_model)
                .filter(|candidate| models::is_installed(candidate))
                .collect()
        };

        // Pick the largest tier whose estimated p90 fits the budget, then
        // verify it. If the real measurement is worse, step down.
        let mut start_index = candidates.iter().position(|candidate| {
            let estimate = timings
                .get(candidate.id)
                .copied()
                .unwrap_or(probe.p90_ms * candidate.relative_cost);
            estimate <= target
        });

        if start_index.is_none() && !candidates.is_empty() {
            start_index = Some(candidates.len() - 1);
            notes.push(format!(
                "No model is estimated to fit the {:.0} ms budget; trying the smallest available model.",
                target
            ));
        }

        if let Some(index) = start_index {
            for candidate in candidates.iter().skip(index) {
                if stop.load(Ordering::Acquire) {
                    return Err(HardwareError::Cancelled);
                }
                let already_measured = timings.get(candidate.id).copied();
                let attempt = if let Some(measured) = already_measured {
                    // The probe itself is already measured and its path is known.
                    if candidate.id == probe_id {
                        Ok((
                            probe_path.clone(),
                            Measurement {
                                p50_ms: probe.p50_ms,
                                p90_ms: measured,
                            },
                        ))
                    } else {
                        ensure_and_measure(candidate, yolo, stop, progress, &mut notes)
                    }
                } else {
                    ensure_and_measure(candidate, yolo, stop, progress, &mut notes)
                };

                match attempt {
                    Ok((path, measurement)) => {
                        timings.insert(candidate.id.to_string(), measurement.p90_ms);
                        let over_budget = measurement.p90_ms > target * 1.20;
                        selected = Some(candidate);
                        selected_path = Some(path);
                        selected_measurement = Some(measurement);
                        selected_id = candidate.id.to_string();
                        if over_budget {
                            notes.push(format!(
                                "{} measured {:.0} ms p90, above the {:.0} ms budget; stepping down.",
                                candidate.id, measurement.p90_ms, target
                            ));
                            continue;
                        }
                        break;
                    }
                    Err(error) => {
                        notes.push(format!("{} is unavailable: {error}", candidate.id));
                    }
                }
            }
        }
    }

    let selected_path = selected_path.unwrap_or(probe_path);
    let selected_measurement = selected_measurement.unwrap_or(probe);
    let selected_id = selected
        .map(|candidate| candidate.id.to_string())
        .unwrap_or(selected_id);

    let report = AutoReport {
        version: BENCHMARK_VERSION,
        fingerprint,
        measured_at: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        hardware,
        target_ms: target,
        probe_id: probe_id.clone(),
        probe_p50_ms: probe.p50_ms,
        probe_p90_ms: probe.p90_ms,
        selected_id: selected_id.clone(),
        selected_model_path: selected_path.display().to_string(),
        selected_p50_ms: Some(selected_measurement.p50_ms),
        selected_p90_ms: Some(selected_measurement.p90_ms),
        timings,
        notes,
        cached: false,
    };

    if let Err(error) = save_cache(&report) {
        tracing::warn!(error = %error, "could not save the hardware selection cache");
    }
    progress(format!(
        "Selected {} (p90 {:.0} ms, target {:.0} ms)",
        report.selected_id,
        report.selected_p90_ms.unwrap_or(0.0),
        report.target_ms
    ));
    Ok(report)
}

fn reuse_cache(
    cache: &mut AutoReport,
    hardware: &HardwareInfo,
    target: f32,
    max_model: ModelSize,
) -> Option<AutoReport> {
    cache.cached = true;
    cache.hardware = hardware.clone();
    cache.target_ms = target;

    let estimated = models::allowed_desc(max_model).find(|candidate| {
        let timing = cache
            .timings
            .get(candidate.id)
            .copied()
            .unwrap_or(cache.probe_p90_ms * candidate.relative_cost);
        timing <= target
    });

    if let Some(candidate) = estimated {
        if models::is_installed(candidate) {
            cache.selected_id = candidate.id.to_string();
            cache.selected_model_path = models::path_for(candidate).display().to_string();
            cache.selected_p50_ms = None;
            cache.selected_p90_ms = cache.timings.get(candidate.id).copied();
            return Some(cache.clone());
        }
    }

    if Path::new(&cache.selected_model_path).is_file() {
        return Some(cache.clone());
    }

    // A cached model may have been removed while the app was upgraded.
    for candidate in models::allowed_asc(max_model) {
        if models::is_installed(candidate) {
            cache.selected_id = candidate.id.to_string();
            cache.selected_model_path = models::path_for(candidate).display().to_string();
            cache.selected_p50_ms = None;
            cache.selected_p90_ms = cache.timings.get(candidate.id).copied();
            return Some(cache.clone());
        }
    }
    if let Some(legacy) = models::legacy_model() {
        cache.selected_id = "legacy".to_string();
        cache.selected_model_path = legacy.display().to_string();
        cache.selected_p50_ms = None;
        cache.selected_p90_ms = Some(cache.probe_p90_ms);
        return Some(cache.clone());
    }
    None
}

fn measure_probe(
    yolo: &YoloConfig,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(String),
    notes: &mut Vec<String>,
) -> Result<(PathBuf, String, Measurement), String> {
    let candidate = models::probe();
    match ensure_and_measure(candidate, yolo, stop, progress, notes) {
        Ok((path, measurement)) => return Ok((path, candidate.id.to_string(), measurement)),
        Err(error) => {
            notes.push(format!(
                "{} could not be used ({error}); trying a local fallback.",
                candidate.id
            ));
        }
    }

    // Prefer the largest installed registry model before the legacy path so
    // an offline install can still use a capable tier.
    for fallback in models::allowed_desc(ModelSize::X) {
        if !models::is_installed(fallback) {
            continue;
        }
        let path = models::path_for(fallback);
        progress(format!("Benchmarking local {}…", fallback.id));
        if let Ok(measurement) = measure(yolo, &path, stop) {
            notes.push(format!(
                "{} measured at {:.0} ms p50 / {:.0} ms p90.",
                fallback.id, measurement.p50_ms, measurement.p90_ms
            ));
            return Ok((path, fallback.id.to_string(), measurement));
        }
    }

    if let Some(path) = models::legacy_model() {
        progress("Benchmarking the local yolo11n model…".to_string());
        let measurement = measure(yolo, &path, stop).map_err(|error| error.to_string())?;
        notes.push(format!(
            "legacy yolo11n measured at {:.0} ms p50 / {:.0} ms p90.",
            measurement.p50_ms, measurement.p90_ms
        ));
        return Ok((path, "legacy".to_string(), measurement));
    }

    Err(notes
        .last()
        .cloned()
        .unwrap_or_else(|| "no model is installed and downloads are unavailable".to_string()))
}

fn ensure_and_measure(
    candidate: &'static ModelCandidate,
    yolo: &YoloConfig,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(String),
    notes: &mut Vec<String>,
) -> Result<(PathBuf, Measurement), HardwareError> {
    let path = models::ensure(candidate, yolo.auto.allow_download, stop, progress)?;
    if stop.load(Ordering::Acquire) {
        return Err(HardwareError::Cancelled);
    }
    progress(format!(
        "Benchmarking {} ({:.0} MB)…",
        candidate.id,
        candidate.bytes as f64 / 1_048_576.0
    ));
    let measurement = match measure(yolo, &path, stop) {
        Ok(measurement) => measurement,
        Err(error) => {
            notes.push(format!("{} failed to load or run: {error}", candidate.id));
            return Err(HardwareError::Benchmark(error.to_string()));
        }
    };
    notes.push(format!(
        "{} measured at {:.0} ms p50 / {:.0} ms p90.",
        candidate.id, measurement.p50_ms, measurement.p90_ms
    ));
    Ok((path, measurement))
}

fn measure(
    yolo: &YoloConfig,
    path: &Path,
    stop: &AtomicBool,
) -> Result<Measurement, HardwareError> {
    let mut config = yolo.clone();
    config.model_path = path.display().to_string();
    let mut engine =
        YoloEngine::load(&config).map_err(|error| HardwareError::Benchmark(error.to_string()))?;
    let frame = benchmark_frame();
    let filter = ClassFilter {
        confidence_threshold: 0.5,
        detect_person: true,
        detect_car: true,
    };

    for _ in 0..WARMUP_RUNS {
        if stop.load(Ordering::Acquire) {
            return Err(HardwareError::Cancelled);
        }
        engine
            .infer(&frame, filter)
            .map_err(|error| HardwareError::Benchmark(error.to_string()))?;
    }

    let started = Instant::now();
    let mut samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        if stop.load(Ordering::Acquire) {
            return Err(HardwareError::Cancelled);
        }
        let run_started = Instant::now();
        engine
            .infer(&frame, filter)
            .map_err(|error| HardwareError::Benchmark(error.to_string()))?;
        samples.push(run_started.elapsed().as_secs_f32() * 1_000.0);
        if started.elapsed() >= BENCHMARK_BUDGET && samples.len() >= 3 {
            break;
        }
    }
    if samples.is_empty() {
        return Err(HardwareError::Benchmark("no benchmark samples".into()));
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Measurement {
        p50_ms: percentile(&samples, 0.50),
        p90_ms: percentile(&samples, 0.90),
    })
}

fn percentile(sorted: &[f32], percentile: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f32 * percentile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn benchmark_frame() -> RgbImage {
    let mut image = RgbImage::new(640, 640);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let r = (x.saturating_mul(3) % 256) as u8;
        let g = (y.saturating_mul(2) % 256) as u8;
        let b = ((x + y).saturating_mul(2) % 256) as u8;
        *pixel = Rgb([r, g, b]);
    }
    // A few solid rectangles give the decoder/NMS realistic work without
    // depending on a copyrighted test photograph.
    for y in 180..460 {
        for x in 210..430 {
            image.put_pixel(x, y, Rgb([210, 60, 40]));
        }
    }
    for y in 300..600 {
        for x in 40..190 {
            image.put_pixel(x, y, Rgb([30, 90, 210]));
        }
    }
    image
}

fn load_cache() -> Option<AutoReport> {
    let bytes = fs::read(CACHE_PATH).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn save_cache(report: &AutoReport) -> Result<(), std::io::Error> {
    if let Some(parent) = Path::new(CACHE_PATH).parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = format!("{CACHE_PATH}.tmp");
    let bytes = serde_json::to_vec_pretty(report).map_err(std::io::Error::other)?;
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, CACHE_PATH)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate;

    fn hardware(memory: u64) -> HardwareInfo {
        HardwareInfo {
            cpu: "Test CPU".into(),
            cores: 4,
            total_memory_mb: Some(memory),
            os: "linux".into(),
            arch: "x86_64".into(),
            device: "cpu".into(),
        }
    }

    #[test]
    fn automatic_budget_is_forty_percent_of_interval() {
        let mut yolo = YoloConfig {
            interval_ms: 1_000,
            ..Default::default()
        };
        assert_eq!(target_ms(&yolo), 400.0);
        yolo.interval_ms = 200;
        assert_eq!(target_ms(&yolo), 100.0);
        yolo.interval_ms = 60_000;
        assert_eq!(target_ms(&yolo), 1_500.0);
        yolo.auto.target_inference_ms = 250;
        assert_eq!(target_ms(&yolo), 250.0);
    }

    #[test]
    fn low_memory_limits_the_largest_model() {
        let mut notes = Vec::new();
        assert_eq!(
            effective_max(&hardware(900), ModelSize::M, &mut notes),
            ModelSize::N
        );
        assert!(!notes.is_empty());
        assert_eq!(
            effective_max(&hardware(8_192), ModelSize::M, &mut notes),
            ModelSize::M
        );
    }

    #[test]
    fn relative_costs_are_ordered() {
        let mut previous = 0.0;
        for candidate in models::allowed_asc(ModelSize::X) {
            assert!(candidate.relative_cost > previous);
            previous = candidate.relative_cost;
        }
    }

    #[test]
    fn fingerprint_includes_device_and_memory() {
        let first = fingerprint(&hardware(8_192));
        let mut other = hardware(8_192);
        other.device = "cuda".into();
        assert_ne!(first, fingerprint(&other));
        let mut more_memory = hardware(16_384);
        more_memory.cpu = "Test CPU".into();
        assert_ne!(first, fingerprint(&more_memory));
    }

    #[test]
    fn probe_is_the_documented_baseline() {
        assert_eq!(models::probe().id, "n640");
        assert_eq!(candidate("n640").unwrap().relative_cost, 1.0);
    }
}
