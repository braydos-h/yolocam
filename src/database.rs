use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection, Row};
use serde::Serialize;
use thiserror::Error;
use tokio::sync::oneshot;

use crate::detection::DetectedClass;

#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("database I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database worker is unavailable")]
    Unavailable,
    #[error("database operation timed out")]
    Timeout,
    #[error("database worker failed: {0}")]
    Worker(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionRecord {
    pub id: i64,
    pub timestamp: String,
    pub epoch_ms: i64,
    pub class: DetectedClass,
    pub confidence: f32,
    pub camera_name: String,
    pub screenshot: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewDetection {
    pub timestamp: String,
    pub epoch_ms: i64,
    pub class: DetectedClass,
    pub confidence: f32,
    pub camera_name: String,
    pub screenshot: Option<String>,
    pub frame_sequence: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HistoryFilter {
    pub class: Option<DetectedClass>,
    pub from_epoch_ms: Option<i64>,
    pub to_epoch_ms: Option<i64>,
    pub limit: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StoredCounts {
    pub people: u64,
    pub cars: u64,
}

enum DbCommand {
    Insert {
        record: NewDetection,
        ack: Option<oneshot::Sender<Result<(), String>>>,
    },
    Query {
        filter: HistoryFilter,
        reply: oneshot::Sender<Result<Vec<DetectionRecord>, String>>,
    },
    Counts {
        reply: oneshot::Sender<Result<StoredCounts, String>>,
    },
    Shutdown {
        ack: oneshot::Sender<()>,
    },
}

#[derive(Clone)]
pub struct Database {
    tx: SyncSender<DbCommand>,
    join: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl Database {
    pub fn open(path: &Path) -> Result<Self, DatabaseError> {
        let mut connection = if path == Path::new(":memory:") {
            Connection::open_in_memory()?
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            Connection::open(path)?
        };
        initialize(&connection)?;

        let (tx, rx) = mpsc::sync_channel::<DbCommand>(1_024);
        let join = std::thread::Builder::new()
            .name("detection-db".to_string())
            .spawn(move || run_worker(&mut connection, rx))
            .map_err(DatabaseError::Io)?;
        Ok(Self {
            tx,
            join: Arc::new(Mutex::new(Some(join))),
        })
    }

    /// Fire-and-forget enqueue from the inference thread. A bounded channel
    /// means a stalled database cannot grow memory without bound.
    pub fn try_enqueue(&self, record: NewDetection) -> bool {
        match self.tx.try_send(DbCommand::Insert { record, ack: None }) {
            Ok(()) => true,
            Err(mpsc::TrySendError::Full(_)) => {
                tracing::warn!("database queue is full; dropping detection event");
                false
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                tracing::warn!("database worker is stopped; dropping detection event");
                false
            }
        }
    }

    pub async fn insert(&self, record: NewDetection) -> Result<(), DatabaseError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .try_send(DbCommand::Insert {
                record,
                ack: Some(ack_tx),
            })
            .map_err(|_| DatabaseError::Unavailable)?;
        match tokio::time::timeout(Duration::from_secs(5), ack_rx).await {
            Ok(Ok(result)) => result.map_err(DatabaseError::Worker),
            Ok(Err(_)) => Err(DatabaseError::Unavailable),
            Err(_) => Err(DatabaseError::Timeout),
        }
    }

    pub async fn history(
        &self,
        filter: HistoryFilter,
    ) -> Result<Vec<DetectionRecord>, DatabaseError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .try_send(DbCommand::Query {
                filter,
                reply: reply_tx,
            })
            .map_err(|_| DatabaseError::Unavailable)?;
        match tokio::time::timeout(Duration::from_secs(5), reply_rx).await {
            Ok(Ok(result)) => result.map_err(DatabaseError::Worker),
            Ok(Err(_)) => Err(DatabaseError::Unavailable),
            Err(_) => Err(DatabaseError::Timeout),
        }
    }

    pub async fn counts(&self) -> Result<StoredCounts, DatabaseError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .try_send(DbCommand::Counts { reply: reply_tx })
            .map_err(|_| DatabaseError::Unavailable)?;
        match tokio::time::timeout(Duration::from_secs(5), reply_rx).await {
            Ok(Ok(result)) => result.map_err(DatabaseError::Worker),
            Ok(Err(_)) => Err(DatabaseError::Unavailable),
            Err(_) => Err(DatabaseError::Timeout),
        }
    }

    pub async fn shutdown(&self) -> Result<(), DatabaseError> {
        let (ack_tx, ack_rx) = oneshot::channel();
        let _ = self.tx.send(DbCommand::Shutdown { ack: ack_tx });
        let _ = tokio::time::timeout(Duration::from_secs(5), ack_rx).await;
        let join = self
            .join
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(join) = join {
            tokio::task::spawn_blocking(move || join.join())
                .await
                .map_err(|error| DatabaseError::Worker(error.to_string()))?
                .map_err(|_| DatabaseError::Worker("database thread panicked".into()))?;
        }
        Ok(())
    }
}

fn initialize(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA busy_timeout = 5000;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch_ms INTEGER NOT NULL,
            class TEXT NOT NULL CHECK (class IN ('person', 'car')),
            class_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            camera_name TEXT NOT NULL,
            screenshot TEXT,
            frame_sequence INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_detections_epoch
            ON detections(epoch_ms DESC);
        CREATE INDEX IF NOT EXISTS idx_detections_class_epoch
            ON detections(class, epoch_ms DESC);
        "#,
    )?;
    Ok(())
}

fn run_worker(connection: &mut Connection, rx: mpsc::Receiver<DbCommand>) {
    while let Ok(command) = rx.recv() {
        match command {
            DbCommand::Insert { record, ack } => {
                let result = insert_record(connection, &record)
                    .map(|_| ())
                    .map_err(|error| error.to_string());
                if let Err(error) = &result {
                    tracing::error!(error = %error, "failed to insert detection event");
                }
                if let Some(ack) = ack {
                    let _ = ack.send(result);
                }
            }
            DbCommand::Query { filter, reply } => {
                let result = query_records(connection, &filter);
                if let Err(error) = &result {
                    tracing::error!(error = %error, "failed to query detection history");
                }
                let _ = reply.send(result);
            }
            DbCommand::Counts { reply } => {
                let result = query_counts(connection);
                if let Err(error) = &result {
                    tracing::error!(error = %error, "failed to query detection counts");
                }
                let _ = reply.send(result);
            }
            DbCommand::Shutdown { ack } => {
                let _ = connection.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                let _ = ack.send(());
                break;
            }
        }
    }
}

fn insert_record(connection: &Connection, record: &NewDetection) -> Result<i64, rusqlite::Error> {
    connection.execute(
        "INSERT INTO detections
            (timestamp, epoch_ms, class, class_id, confidence, camera_name, screenshot, frame_sequence)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            record.timestamp,
            record.epoch_ms,
            record.class.as_str(),
            record.class.coco_id() as i64,
            f64::from(record.confidence),
            record.camera_name,
            record.screenshot,
            record.frame_sequence as i64,
        ],
    )?;
    Ok(connection.last_insert_rowid())
}

fn query_records(
    connection: &Connection,
    filter: &HistoryFilter,
) -> Result<Vec<DetectionRecord>, String> {
    let mut sql = String::from(
        "SELECT id, timestamp, epoch_ms, class, confidence, camera_name, screenshot
         FROM detections",
    );
    let mut clauses: Vec<&str> = Vec::new();
    let mut values: Vec<Value> = Vec::new();

    if let Some(class) = filter.class {
        clauses.push("class = ?");
        values.push(Value::Text(class.as_str().to_string()));
    }
    if let Some(from) = filter.from_epoch_ms {
        clauses.push("epoch_ms >= ?");
        values.push(Value::Integer(from));
    }
    if let Some(to) = filter.to_epoch_ms {
        clauses.push("epoch_ms <= ?");
        values.push(Value::Integer(to));
    }
    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" AND "));
    }
    sql.push_str(" ORDER BY epoch_ms DESC, id DESC LIMIT ? OFFSET ?");
    values.push(Value::Integer(i64::from(filter.limit.clamp(1, 1_000))));
    values.push(Value::Integer(i64::from(filter.offset)));

    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(params_from_iter(values.iter()), map_record)
        .map_err(|error| error.to_string())?;
    let mut records = Vec::new();
    for row in rows {
        records.push(row.map_err(|error| error.to_string())?);
    }
    Ok(records)
}

fn map_record(row: &Row<'_>) -> rusqlite::Result<DetectionRecord> {
    let class_name: String = row.get(3)?;
    let class = DetectedClass::from_name(&class_name).ok_or_else(|| {
        rusqlite::Error::InvalidColumnType(3, "class".to_string(), rusqlite::types::Type::Text)
    })?;
    Ok(DetectionRecord {
        id: row.get(0)?,
        timestamp: row.get(1)?,
        epoch_ms: row.get(2)?,
        class,
        confidence: row.get::<_, f64>(4)? as f32,
        camera_name: row.get(5)?,
        screenshot: row.get(6)?,
    })
}

fn query_counts(connection: &Connection) -> Result<StoredCounts, String> {
    let mut statement = connection
        .prepare("SELECT class, COUNT(*) FROM detections GROUP BY class")
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let class: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((class, count))
        })
        .map_err(|error| error.to_string())?;
    let mut counts = StoredCounts::default();
    for row in rows {
        let (class, count) = row.map_err(|error| error.to_string())?;
        let count = count.max(0) as u64;
        match class.as_str() {
            "person" => counts.people = count,
            "car" => counts.cars = count,
            _ => {}
        }
    }
    Ok(counts)
}

pub fn database_path_for_display(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn record(class: DetectedClass, epoch_ms: i64) -> NewDetection {
        NewDetection {
            timestamp: chrono::DateTime::<chrono::Utc>::from_timestamp_millis(epoch_ms)
                .unwrap()
                .to_rfc3339(),
            epoch_ms,
            class,
            confidence: 0.9,
            camera_name: "Front Camera".to_string(),
            screenshot: Some("front-camera_2026-11-23_19-30-01-123.jpg".to_string()),
            frame_sequence: 42,
        }
    }

    #[tokio::test]
    async fn inserts_and_filters_records() {
        let dir = tempdir().unwrap();
        let db = Database::open(&dir.path().join("test.sqlite3")).unwrap();
        db.insert(record(DetectedClass::Person, 1_000))
            .await
            .unwrap();
        db.insert(record(DetectedClass::Car, 2_000)).await.unwrap();
        db.insert(record(DetectedClass::Person, 3_000))
            .await
            .unwrap();

        let all = db
            .history(HistoryFilter {
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].epoch_ms, 3_000);

        let people = db
            .history(HistoryFilter {
                class: Some(DetectedClass::Person),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(people.len(), 2);
        assert!(people
            .iter()
            .all(|record| record.class == DetectedClass::Person));

        let window = db
            .history(HistoryFilter {
                from_epoch_ms: Some(1_500),
                to_epoch_ms: Some(3_500),
                limit: 10,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(window.len(), 2);

        let counts = db.counts().await.unwrap();
        assert_eq!(counts.people, 2);
        assert_eq!(counts.cars, 1);
        db.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn rejects_database_from_worker_when_queue_full_is_bounded() {
        let db = Database::open(Path::new(":memory:")).unwrap();
        // Ensure normal enqueue works, then shut the worker down and verify
        // that new events are rejected instead of blocking or panicking.
        assert!(db.try_enqueue(record(DetectedClass::Person, 1)));
        db.shutdown().await.unwrap();
        assert!(!db.try_enqueue(record(DetectedClass::Person, 2)));
    }
}
