use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::ModelSize;

/// Models selected by the hardware benchmark live here so an explicitly
/// configured model is never overwritten.
pub const AUTO_MODEL_DIR: &str = "models/auto";

const REGISTRY_REVISION: &str = "8b180c762d5cf217886d16e64403508042207e4d";
const REGISTRY_BASE: &str = "https://huggingface.co/giangndm/yolo11-onnx/resolve";
const MAX_DOWNLOAD_OVERHEAD: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model {0} is not installed and automatic downloads are disabled")]
    DownloadDisabled(&'static str),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("model download was interrupted")]
    Cancelled,
    #[error("model file I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("model {0} failed verification (expected {1} bytes, SHA-256 {2})")]
    Verification(&'static str, u64, &'static str),
}

#[derive(Clone, Copy, Debug)]
pub struct ModelCandidate {
    pub id: &'static str,
    pub file: &'static str,
    pub size: ModelSize,
    pub bytes: u64,
    pub sha256: &'static str,
    /// Approximate p90 inference cost relative to yolo11n at 640x640.
    pub relative_cost: f32,
}

/// Ordered from smallest/fastest to largest. The first entry is the emergency
/// fallback for machines that cannot sustain yolo11n at 640x640.
pub const CANDIDATES: &[ModelCandidate] = &[
    ModelCandidate {
        id: "n320",
        file: "yolo11n_320.onnx",
        size: ModelSize::N,
        bytes: 10_594_049,
        sha256: "ed25687f7f3176a041bcdd0173501629afcaaddf37711f2294ced377f249fd65",
        relative_cost: 0.32,
    },
    ModelCandidate {
        id: "n640",
        file: "yolo11n_640.onnx",
        size: ModelSize::N,
        bytes: 10_720_104,
        sha256: "71b891f38f484175c65b0277612e0305789bad9bd88b4e2def7609f18a23836f",
        relative_cost: 1.0,
    },
    ModelCandidate {
        id: "s640",
        file: "yolo11s_640.onnx",
        size: ModelSize::S,
        bytes: 38_030_388,
        sha256: "20a6bd8ba7c291e688a48233e35584284bff04decdd077193e5f60d2b7dd6f39",
        relative_cost: 2.6,
    },
    ModelCandidate {
        id: "m640",
        file: "yolo11m_640.onnx",
        size: ModelSize::M,
        bytes: 80_646_471,
        sha256: "559addb423c62f9c449aeb08e0ec4ffc23b43a0f50d0b3275108acdd6e00c893",
        relative_cost: 6.2,
    },
    ModelCandidate {
        id: "l640",
        file: "yolo11l_640.onnx",
        size: ModelSize::L,
        bytes: 101_702_593,
        sha256: "ef67dba3126080e87d08ba58dc4e1a639dc77a8a69982064ebb12007883cde71",
        relative_cost: 8.8,
    },
    ModelCandidate {
        id: "x640",
        file: "yolo11x_640.onnx",
        size: ModelSize::X,
        bytes: 228_016_347,
        sha256: "e8e890f0ee718ae277adba400c4597317b072b667563062d57c2dbacf60d25e5",
        relative_cost: 17.0,
    },
];

#[derive(Debug, Clone, Serialize)]
pub struct CandidateView {
    pub id: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    pub installed: bool,
    pub allowed: bool,
}

pub fn candidate(id: &str) -> Option<&'static ModelCandidate> {
    CANDIDATES.iter().find(|candidate| candidate.id == id)
}

pub fn probe() -> &'static ModelCandidate {
    candidate("n640").expect("n640 probe is in the registry")
}

pub fn allowed(candidate: &ModelCandidate, max_model: ModelSize) -> bool {
    // n320 is always allowed: it is the emergency tier for weak hardware.
    candidate.id == "n320" || candidate.size.rank() <= max_model.rank()
}

pub fn allowed_desc(max_model: ModelSize) -> impl Iterator<Item = &'static ModelCandidate> {
    CANDIDATES
        .iter()
        .rev()
        .filter(move |candidate| allowed(candidate, max_model))
}

pub fn allowed_asc(max_model: ModelSize) -> impl Iterator<Item = &'static ModelCandidate> {
    CANDIDATES
        .iter()
        .filter(move |candidate| allowed(candidate, max_model))
}

pub fn path_for(candidate: &ModelCandidate) -> PathBuf {
    Path::new(AUTO_MODEL_DIR).join(candidate.file)
}

pub fn is_installed(candidate: &ModelCandidate) -> bool {
    fs::metadata(path_for(candidate))
        .map(|metadata| metadata.len() == candidate.bytes)
        .unwrap_or(false)
}

/// The model from earlier releases, used as a last-resort fallback when the
/// automatic registry is unavailable (for example on an offline install).
pub fn legacy_model() -> Option<PathBuf> {
    let path = PathBuf::from("models/yolo11n.onnx");
    path.is_file().then_some(path)
}

pub fn public_candidates(max_model: ModelSize) -> Vec<CandidateView> {
    CANDIDATES
        .iter()
        .map(|candidate| CandidateView {
            id: candidate.id,
            file: candidate.file,
            bytes: candidate.bytes,
            installed: is_installed(candidate),
            allowed: allowed(candidate, max_model),
        })
        .collect()
}

/// Return a verified local path for a candidate, downloading it first when
/// permitted. Existing files are checked by size and only re-hashed after a
/// download; hashing a 200 MB model on every start would be wasteful.
pub fn ensure(
    candidate: &'static ModelCandidate,
    allow_download: bool,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(String),
) -> Result<PathBuf, ModelError> {
    let path = path_for(candidate);
    if is_installed(candidate) {
        return Ok(path);
    }
    if !allow_download {
        return Err(ModelError::DownloadDisabled(candidate.id));
    }

    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory)?;
    // Windows will not rename over an existing file; a wrong-sized file is
    // unusable anyway, so replace it.
    if path.exists() {
        let _ = fs::remove_file(&path);
    }
    let temporary = path.with_extension("onnx.part");
    let _ = fs::remove_file(&temporary);

    let url = format!("{REGISTRY_BASE}/{REGISTRY_REVISION}/{}", candidate.file);
    progress(format!(
        "Downloading {} ({:.1} MB)",
        candidate.file,
        candidate.bytes as f64 / 1_048_576.0
    ));

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();
    let response = agent
        .get(&url)
        .call()
        .map_err(|error| ModelError::Download(error.to_string()))?;
    let mut reader = response.into_reader();
    let mut file = File::create(&temporary)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut next_report: u64 = 8 * 1024 * 1024;

    loop {
        if stop.load(Ordering::Acquire) {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(ModelError::Cancelled);
        }
        let read = reader
            .read(&mut buffer)
            .map_err(|error| ModelError::Download(error.to_string()))?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > candidate.bytes.saturating_add(MAX_DOWNLOAD_OVERHEAD) {
            drop(file);
            let _ = fs::remove_file(&temporary);
            return Err(ModelError::Download(format!(
                "{} is larger than the expected {} bytes",
                candidate.file, candidate.bytes
            )));
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])?;
        if total >= next_report {
            progress(format!(
                "Downloading {}: {:.1} / {:.1} MB",
                candidate.file,
                total as f64 / 1_048_576.0,
                candidate.bytes as f64 / 1_048_576.0
            ));
            next_report = total.saturating_add(8 * 1024 * 1024);
        }
    }
    file.sync_all()?;
    drop(file);

    let actual: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if total != candidate.bytes || actual != candidate.sha256 {
        let _ = fs::remove_file(&temporary);
        return Err(ModelError::Verification(
            candidate.id,
            candidate.bytes,
            candidate.sha256,
        ));
    }

    fs::rename(&temporary, &path)?;
    progress(format!("Verified {}", candidate.file));
    Ok(path)
}

/// Remove a partially downloaded model after a failed verification.
#[allow(dead_code)]
pub fn discard(candidate: &ModelCandidate) {
    let _ = fs::remove_file(path_for(candidate));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_ordered_and_unique() {
        let mut previous = 0.0_f32;
        let mut ids = std::collections::HashSet::new();
        for candidate in CANDIDATES {
            assert!(candidate.relative_cost > previous);
            assert!(ids.insert(candidate.id));
            assert!(candidate.bytes > 0);
            assert!(candidate.sha256.len() == 64);
            previous = candidate.relative_cost;
        }
    }

    #[test]
    fn n320_is_allowed_even_when_max_is_n() {
        let n320 = candidate("n320").unwrap();
        let n640 = candidate("n640").unwrap();
        assert!(allowed(n320, ModelSize::N));
        assert!(allowed(n640, ModelSize::N));
        assert!(!allowed(candidate("s640").unwrap(), ModelSize::N));
    }

    #[test]
    fn max_model_filters_larger_tiers() {
        assert_eq!(
            allowed_desc(ModelSize::M)
                .map(|candidate| candidate.id)
                .collect::<Vec<_>>(),
            vec!["m640", "s640", "n640", "n320"]
        );
    }
}
