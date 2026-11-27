use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use crate::certificate::{
    ChainAuditSummary, ZeroizationCertificate, CERTIFICATE_SCHEMA_V2, GENESIS_PREV_CERT_SHA256,
};
use crate::error::ScrubError;

/// Distinct CLI exit code emitted when a device is newly or previously quarantined.
pub const EXIT_CODE_QUARANTINED: u8 = 42;

const LOCK_MAX_ATTEMPTS: usize = 500;
const LOCK_SLEEP_MS: u64 = 20;
const STALE_LOCK_SECS: u64 = 120;
const QUARANTINE_SCHEMA_V1: &str = "zervram.quarantine.v1";

pub type BuildAndSignFn =
    Box<dyn FnOnce(String) -> Result<ZeroizationCertificate, ScrubError> + Send>;

/// Returns the platform-appropriate persistent Zervram state directory:
/// - Windows: `%LOCALAPPDATA%\zervram`
/// - Unix/Linux: `$XDG_DATA_HOME/zervram` or `$HOME/.local/share/zervram` (fallback `/var/lib/zervram`)
pub fn default_state_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            if !local.is_empty() {
                return PathBuf::from(local).join("zervram");
            }
        }
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            if !profile.is_empty() {
                return PathBuf::from(profile)
                    .join("AppData")
                    .join("Local")
                    .join("zervram");
            }
        }
        PathBuf::from(".").join("zervram")
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            if !xdg.is_empty() {
                return PathBuf::from(xdg).join("zervram");
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            if !home.is_empty() {
                return PathBuf::from(home)
                    .join(".local")
                    .join("share")
                    .join("zervram");
            }
        }
        PathBuf::from("/var/lib/zervram")
    }
}

pub fn default_chain_file_path() -> PathBuf {
    default_state_dir().join("chain.jsonl")
}

pub fn default_quarantine_file_path(chain_file: Option<&Path>) -> PathBuf {
    if let Some(cf) = chain_file {
        if let Some(parent) = cf.parent() {
            if !parent.as_os_str().is_empty() {
                return parent.join("quarantine.json");
            }
        }
    }
    default_state_dir().join("quarantine.json")
}

fn sidecar_lock_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_os_string();
    os.push(".lock");
    PathBuf::from(os)
}

/// Cross-platform pure-Rust atomic lockfile guard (`O_CREAT | O_EXCL` / `CREATE_NEW`) requiring
/// zero `unsafe` code. Drops its file handle immediately after creation so Windows can cleanly
/// unlink the lock file on drop.
struct FileLockGuard {
    lock_path: PathBuf,
}

impl FileLockGuard {
    fn acquire(lock_path: &Path) -> Result<Self, ScrubError> {
        if let Some(parent) = lock_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    ScrubError::StorageFailure(format!(
                        "failed to create lock directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }

        for _ in 0..LOCK_MAX_ATTEMPTS {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
            {
                Ok(mut file) => {
                    let _ = writeln!(file, "pid={}", std::process::id());
                    let _ = file.sync_all();
                    drop(file);
                    return Ok(Self {
                        lock_path: lock_path.to_path_buf(),
                    });
                }
                Err(err)
                    if err.kind() == ErrorKind::AlreadyExists
                        || err.kind() == ErrorKind::PermissionDenied =>
                {
                    if let Ok(meta) = std::fs::metadata(lock_path) {
                        if let Ok(modified) = meta.modified() {
                            if let Ok(age) = SystemTime::now().duration_since(modified) {
                                if age > Duration::from_secs(STALE_LOCK_SECS) {
                                    let _ = std::fs::remove_file(lock_path);
                                    continue;
                                }
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_millis(LOCK_SLEEP_MS));
                }
                Err(err) => {
                    return Err(ScrubError::StorageFailure(format!(
                        "failed to acquire lock '{}': {err}",
                        lock_path.display()
                    )));
                }
            }
        }

        Err(ScrubError::StorageFailure(format!(
            "timed out waiting for lock '{}'",
            lock_path.display()
        )))
    }
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

#[async_trait]
pub trait CertificateStore: Send + Sync + 'static {
    /// Atomically links a new certificate to the current chain head, signs via `build_and_sign`,
    /// persists it, and returns `(certificate, new_cert_sha256)`.
    async fn append_signed(
        &self,
        build_and_sign: BuildAndSignFn,
    ) -> Result<(ZeroizationCertificate, String), ScrubError>;

    async fn latest_hash(&self) -> Result<String, ScrubError>;

    async fn get_by_id(&self, cert_id: Uuid) -> Result<Option<ZeroizationCertificate>, ScrubError>;

    async fn get_with_predecessor(
        &self,
        cert_id: Uuid,
    ) -> Result<Option<(ZeroizationCertificate, Option<ZeroizationCertificate>)>, ScrubError>;

    async fn list_recent(&self, limit: usize) -> Result<Vec<ZeroizationCertificate>, ScrubError>;

    async fn total_count(&self) -> Result<usize, ScrubError>;

    async fn verify_full_chain(
        &self,
        verifying_key: &[u8; 32],
    ) -> Result<ChainAuditSummary, ScrubError>;
}

// ------------------------------------------------------------------------------
// InMemoryCertificateStore (for unit tests & ephemeral examples)
// ------------------------------------------------------------------------------

#[derive(Debug)]
struct StoreInner {
    chain: Vec<ZeroizationCertificate>,
    index_by_id: HashMap<Uuid, usize>,
    head_sha256: String,
}

#[derive(Debug)]
pub struct InMemoryCertificateStore {
    inner: RwLock<StoreInner>,
}

impl Default for InMemoryCertificateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCertificateStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(StoreInner {
                chain: Vec::new(),
                index_by_id: HashMap::new(),
                head_sha256: GENESIS_PREV_CERT_SHA256.to_string(),
            }),
        }
    }
}

#[async_trait]
impl CertificateStore for InMemoryCertificateStore {
    async fn append_signed(
        &self,
        build_and_sign: BuildAndSignFn,
    ) -> Result<(ZeroizationCertificate, String), ScrubError> {
        let mut guard = self.inner.write().await;
        let expected_prev = guard.head_sha256.clone();
        let cert = build_and_sign(expected_prev.clone())?;

        if cert.prev_cert_sha256 != expected_prev {
            return Err(ScrubError::ChainIntegrityViolation {
                cert_id: cert.cert_id,
                expected: expected_prev,
                actual: cert.prev_cert_sha256,
            });
        }

        let new_hash = cert.certificate_sha256_hex()?;
        let next_idx = guard.chain.len();
        guard.index_by_id.insert(cert.cert_id, next_idx);
        guard.chain.push(cert.clone());
        guard.head_sha256 = new_hash.clone();

        Ok((cert, new_hash))
    }

    async fn latest_hash(&self) -> Result<String, ScrubError> {
        let guard = self.inner.read().await;
        Ok(guard.head_sha256.clone())
    }

    async fn get_by_id(&self, cert_id: Uuid) -> Result<Option<ZeroizationCertificate>, ScrubError> {
        let guard = self.inner.read().await;
        let maybe_idx = guard.index_by_id.get(&cert_id).copied();
        Ok(maybe_idx.and_then(|idx| guard.chain.get(idx).cloned()))
    }

    async fn get_with_predecessor(
        &self,
        cert_id: Uuid,
    ) -> Result<Option<(ZeroizationCertificate, Option<ZeroizationCertificate>)>, ScrubError> {
        let guard = self.inner.read().await;
        let Some(idx) = guard.index_by_id.get(&cert_id).copied() else {
            return Ok(None);
        };
        let Some(cert) = guard.chain.get(idx).cloned() else {
            return Ok(None);
        };
        let predecessor = if idx == 0 {
            None
        } else {
            guard.chain.get(idx.saturating_sub(1)).cloned()
        };
        Ok(Some((cert, predecessor)))
    }

    async fn list_recent(&self, limit: usize) -> Result<Vec<ZeroizationCertificate>, ScrubError> {
        let guard = self.inner.read().await;
        Ok(guard.chain.iter().rev().take(limit).cloned().collect())
    }

    async fn total_count(&self) -> Result<usize, ScrubError> {
        let guard = self.inner.read().await;
        Ok(guard.chain.len())
    }

    async fn verify_full_chain(
        &self,
        verifying_key_bytes: &[u8; 32],
    ) -> Result<ChainAuditSummary, ScrubError> {
        let verifying_key = VerifyingKey::from_bytes(verifying_key_bytes)
            .map_err(|err| ScrubError::SigningFailure(err.to_string()))?;

        let guard = self.inner.read().await;
        let mut expected_prev = GENESIS_PREV_CERT_SHA256.to_string();
        let mut all_signatures_valid = true;
        let mut chain_intact = true;

        for cert in &guard.chain {
            if cert.schema_version != CERTIFICATE_SCHEMA_V2 || cert.prev_cert_sha256 != expected_prev
            {
                chain_intact = false;
            }
            if cert.verify_signature(&verifying_key).is_err() {
                all_signatures_valid = false;
            }
            expected_prev = cert.certificate_sha256_hex()?;
        }

        if expected_prev != guard.head_sha256 {
            chain_intact = false;
        }

        Ok(ChainAuditSummary {
            total_certificates: guard.chain.len(),
            head_sha256: guard.head_sha256.clone(),
            all_signatures_valid,
            chain_intact,
        })
    }
}

// ------------------------------------------------------------------------------
// AppendOnlyChainStore (Persistent, cross-process file-locked JSONL + sync_all)
// ------------------------------------------------------------------------------

/// File-locked, append-only JSONL certificate store with `sync_all()` durability.
/// Fails closed if the persisted chain head cannot be read or contains broken hash links.
#[derive(Debug)]
pub struct AppendOnlyChainStore {
    path: PathBuf,
    lock_path: PathBuf,
    in_process_lock: Mutex<()>,
}

impl AppendOnlyChainStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, ScrubError> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    ScrubError::StorageFailure(format!(
                        "failed to create chain directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }
        let lock_path = sidecar_lock_path(&path);
        let store = Self {
            path,
            lock_path,
            in_process_lock: Mutex::new(()),
        };
        // Validate existing chain on open (fail closed on malformed JSON or broken links).
        let _lock = FileLockGuard::acquire(&store.lock_path)?;
        let _ = store.read_verified_chain_locked()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_raw_records_locked(&self) -> Result<Vec<ZeroizationCertificate>, ScrubError> {
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => {
                return Err(ScrubError::StorageFailure(format!(
                    "failed to open chain file '{}': {err}",
                    self.path.display()
                )));
            }
        };

        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for (line_idx, line_res) in reader.lines().enumerate() {
            let line = line_res.map_err(|err| {
                ScrubError::StorageFailure(format!(
                    "failed to read line {} of '{}': {err}",
                    line_idx + 1,
                    self.path.display()
                ))
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let cert: ZeroizationCertificate = serde_json::from_str(trimmed).map_err(|err| {
                ScrubError::StorageFailure(format!(
                    "corrupt certificate JSON at line {} of '{}': {err}",
                    line_idx + 1,
                    self.path.display()
                ))
            })?;
            records.push(cert);
        }
        Ok(records)
    }

    fn read_verified_chain_locked(
        &self,
    ) -> Result<(Vec<ZeroizationCertificate>, String), ScrubError> {
        let records = self.read_raw_records_locked()?;
        let mut expected_prev = GENESIS_PREV_CERT_SHA256.to_string();
        for cert in &records {
            if cert.prev_cert_sha256 != expected_prev {
                return Err(ScrubError::ChainIntegrityViolation {
                    cert_id: cert.cert_id,
                    expected: expected_prev,
                    actual: cert.prev_cert_sha256.clone(),
                });
            }
            expected_prev = cert.certificate_sha256_hex()?;
        }
        Ok((records, expected_prev))
    }

    /// Validates every certificate in the persistent JSONL chain against a set of pinned
    /// verifying keys (supporting `--pubkey`, `--key-file`, or `--trust-dir`).
    pub async fn verify_full_chain_with_keys(
        &self,
        trusted_keys: &[VerifyingKey],
    ) -> Result<ChainAuditSummary, ScrubError> {
        if trusted_keys.is_empty() {
            return Err(ScrubError::VerificationFailure(
                "verify-chain requires at least one pinned trusted Ed25519 verifying key"
                    .to_string(),
            ));
        }

        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let records = self.read_raw_records_locked()?;

        let mut expected_prev = GENESIS_PREV_CERT_SHA256.to_string();
        let mut all_signatures_valid = true;
        let mut chain_intact = true;

        for cert in &records {
            if cert.schema_version != CERTIFICATE_SCHEMA_V2 || cert.prev_cert_sha256 != expected_prev
            {
                chain_intact = false;
            }
            let sig_ok = trusted_keys
                .iter()
                .any(|vk| cert.verify_signature(vk).is_ok());
            if !sig_ok {
                all_signatures_valid = false;
            }
            expected_prev = cert.certificate_sha256_hex()?;
        }

        Ok(ChainAuditSummary {
            total_certificates: records.len(),
            head_sha256: expected_prev,
            all_signatures_valid,
            chain_intact,
        })
    }
}

#[async_trait]
impl CertificateStore for AppendOnlyChainStore {
    async fn append_signed(
        &self,
        build_and_sign: BuildAndSignFn,
    ) -> Result<(ZeroizationCertificate, String), ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;

        let (_existing, expected_prev) = self.read_verified_chain_locked()?;
        let cert = build_and_sign(expected_prev.clone())?;

        if cert.prev_cert_sha256 != expected_prev {
            return Err(ScrubError::ChainIntegrityViolation {
                cert_id: cert.cert_id,
                expected: expected_prev,
                actual: cert.prev_cert_sha256,
            });
        }

        let new_hash = cert.certificate_sha256_hex()?;
        let serialized = serde_json::to_string(&cert).map_err(|err| {
            ScrubError::StorageFailure(format!("failed to serialize certificate JSONL: {err}"))
        })?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|err| {
                ScrubError::StorageFailure(format!(
                    "failed to open chain file '{}' for append: {err}",
                    self.path.display()
                ))
            })?;

        writeln!(file, "{serialized}").map_err(|err| {
            ScrubError::StorageFailure(format!(
                "failed to append certificate to '{}': {err}",
                self.path.display()
            ))
        })?;
        file.flush().map_err(|err| {
            ScrubError::StorageFailure(format!(
                "failed to flush chain file '{}': {err}",
                self.path.display()
            ))
        })?;
        file.sync_all().map_err(|err| {
            ScrubError::StorageFailure(format!(
                "failed to fsync chain file '{}': {err}",
                self.path.display()
            ))
        })?;

        Ok((cert, new_hash))
    }

    async fn latest_hash(&self) -> Result<String, ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let (_records, head) = self.read_verified_chain_locked()?;
        Ok(head)
    }

    async fn get_by_id(&self, cert_id: Uuid) -> Result<Option<ZeroizationCertificate>, ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let records = self.read_raw_records_locked()?;
        Ok(records.into_iter().find(|c| c.cert_id == cert_id))
    }

    async fn get_with_predecessor(
        &self,
        cert_id: Uuid,
    ) -> Result<Option<(ZeroizationCertificate, Option<ZeroizationCertificate>)>, ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let records = self.read_raw_records_locked()?;
        let Some(pos) = records.iter().position(|c| c.cert_id == cert_id) else {
            return Ok(None);
        };
        let Some(cert) = records.get(pos).cloned() else {
            return Ok(None);
        };
        let predecessor = if pos == 0 {
            None
        } else {
            records.get(pos.saturating_sub(1)).cloned()
        };
        Ok(Some((cert, predecessor)))
    }

    async fn list_recent(&self, limit: usize) -> Result<Vec<ZeroizationCertificate>, ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let records = self.read_raw_records_locked()?;
        Ok(records.into_iter().rev().take(limit).collect())
    }

    async fn total_count(&self) -> Result<usize, ScrubError> {
        let _proc_guard = self.in_process_lock.lock().await;
        let _file_lock = FileLockGuard::acquire(&self.lock_path)?;
        let records = self.read_raw_records_locked()?;
        Ok(records.len())
    }

    async fn verify_full_chain(
        &self,
        verifying_key_bytes: &[u8; 32],
    ) -> Result<ChainAuditSummary, ScrubError> {
        let verifying_key = VerifyingKey::from_bytes(verifying_key_bytes)
            .map_err(|err| ScrubError::SigningFailure(err.to_string()))?;
        self.verify_full_chain_with_keys(&[verifying_key]).await
    }
}

// ------------------------------------------------------------------------------
// QuarantineStore (Persistent quarantine.json surviving across CLI invocations)
// ------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantineRecord {
    pub device_uuid: Uuid,
    pub pci_bus_id: String,
    pub vendor_model: String,
    pub reason: String,
    pub quarantined_at: DateTime<Utc>,
    pub failed_cert_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineStateFile {
    pub schema_version: String,
    pub devices: BTreeMap<Uuid, QuarantineRecord>,
}

impl Default for QuarantineStateFile {
    fn default() -> Self {
        Self {
            schema_version: QUARANTINE_SCHEMA_V1.to_string(),
            devices: BTreeMap::new(),
        }
    }
}

/// File-locked persistent quarantine registry (`quarantine.json`) so faulted devices remain
/// blocked across CLI invocations and node restarts.
#[derive(Debug)]
pub struct QuarantineStore {
    path: PathBuf,
    lock_path: PathBuf,
}

impl QuarantineStore {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let lock_path = sidecar_lock_path(&path);
        Self { path, lock_path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_state_locked(&self) -> Result<QuarantineStateFile, ScrubError> {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(s) => s,
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Ok(QuarantineStateFile::default());
            }
            Err(err) => {
                return Err(ScrubError::StorageFailure(format!(
                    "failed to read quarantine state '{}': {err}",
                    self.path.display()
                )));
            }
        };
        serde_json::from_str(&raw).map_err(|err| {
            ScrubError::StorageFailure(format!(
                "corrupt quarantine state '{}': {err}",
                self.path.display()
            ))
        })
    }

    fn write_state_locked(&self, state: &QuarantineStateFile) -> Result<(), ScrubError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    ScrubError::StorageFailure(format!(
                        "failed to create quarantine directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }

        let payload = serde_json::to_string_pretty(state).map_err(|err| {
            ScrubError::StorageFailure(format!("failed to serialize quarantine state: {err}"))
        })?;

        let mut tmp_os = self.path.as_os_str().to_os_string();
        tmp_os.push(format!(".tmp.{}", std::process::id()));
        let tmp_path = PathBuf::from(tmp_os);

        let mut tmp_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|err| {
                ScrubError::StorageFailure(format!(
                    "failed to create temporary quarantine file '{}': {err}",
                    tmp_path.display()
                ))
            })?;
        tmp_file.write_all(payload.as_bytes()).map_err(|err| {
            ScrubError::StorageFailure(format!("failed to write quarantine state: {err}"))
        })?;
        tmp_file.sync_all().map_err(|err| {
            ScrubError::StorageFailure(format!("failed to fsync quarantine state: {err}"))
        })?;
        drop(tmp_file);

        #[cfg(target_os = "windows")]
        {
            if self.path.exists() {
                std::fs::remove_file(&self.path).map_err(|err| {
                    ScrubError::StorageFailure(format!(
                        "failed to replace quarantine file '{}': {err}",
                        self.path.display()
                    ))
                })?;
            }
        }

        std::fs::rename(&tmp_path, &self.path).map_err(|err| {
            ScrubError::StorageFailure(format!(
                "failed to commit quarantine state '{}': {err}",
                self.path.display()
            ))
        })?;
        Ok(())
    }

    pub fn is_quarantined(&self, device_uuid: Uuid) -> Result<Option<QuarantineRecord>, ScrubError> {
        let _lock = FileLockGuard::acquire(&self.lock_path)?;
        let state = self.read_state_locked()?;
        Ok(state.devices.get(&device_uuid).cloned())
    }

    pub fn record_quarantine(&self, record: QuarantineRecord) -> Result<(), ScrubError> {
        let _lock = FileLockGuard::acquire(&self.lock_path)?;
        let mut state = self.read_state_locked()?;
        state.devices.insert(record.device_uuid, record);
        self.write_state_locked(&state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{hash_tenant_id, ZeroizationCertificateV2};
    use crate::signer::{AttestationSigner, Ed25519Signer};
    use zervram_hal::{BackendKind, VerificationMode};

    fn make_cert(signer: &Ed25519Signer, prev: String) -> Result<ZeroizationCertificate, ScrubError> {
        let mut cert = ZeroizationCertificateV2 {
            schema_version: CERTIFICATE_SCHEMA_V2.to_string(),
            cert_id: Uuid::now_v7(),
            device_uuid: Uuid::new_v4(),
            device_ordinal: 0,
            pci_bus_id: "0000:00:04.0".to_string(),
            vendor_model: "Tesla T4".to_string(),
            backend_kind: BackendKind::MockCpu,
            tenant_prev_hash: hash_tenant_id("tenant-persist"),
            vram_total_bytes: 16 * 1024 * 1024 * 1024,
            vram_free_bytes: 15 * 1024 * 1024 * 1024,
            swept_bytes: 524_288,
            chunk_count: 1,
            chunk_table_sha256: "b".repeat(64),
            verification_mode: VerificationMode::MockAllMapped,
            verified_bytes: 524_288,
            sampled_pages: 128,
            active_processes_before: 0,
            challenge_nonce: None,
            duration_us: 900,
            verified_zero: true,
            timestamp_utc: Utc::now(),
            prev_cert_sha256: prev,
            ed25519_signature_hex: String::new(),
        };
        let digest = cert.canonical_signing_digest()?;
        cert.ed25519_signature_hex = hex::encode(signer.sign_digest(&digest)?);
        Ok(cert)
    }

    #[tokio::test]
    async fn append_only_store_chains_across_separate_open_instances() -> Result<(), ScrubError> {
        let dir = std::env::temp_dir().join(format!("zervram_chain_test_{}", Uuid::new_v4()));
        let chain_path = dir.join("chain.jsonl");
        let signer = Ed25519Signer::generate_ephemeral();

        // Run 1
        let (cert1, sha1) = {
            let store1 = AppendOnlyChainStore::open(&chain_path)?;
            let s_key = signer.clone_signing_key();
            store1
                .append_signed(Box::new(move |prev| {
                    let s = Ed25519Signer::from_signing_key(s_key);
                    make_cert(&s, prev)
                }))
                .await?
        };
        assert_eq!(cert1.prev_cert_sha256, GENESIS_PREV_CERT_SHA256);

        // Run 2 (fresh store handle reading from disk)
        let (cert2, sha2) = {
            let store2 = AppendOnlyChainStore::open(&chain_path)?;
            let s_key = signer.clone_signing_key();
            store2
                .append_signed(Box::new(move |prev| {
                    let s = Ed25519Signer::from_signing_key(s_key);
                    make_cert(&s, prev)
                }))
                .await?
        };
        assert_eq!(cert2.prev_cert_sha256, sha1);

        let verifier_store = AppendOnlyChainStore::open(&chain_path)?;
        let summary = verifier_store
            .verify_full_chain(&signer.verifying_key_bytes())
            .await?;
        assert_eq!(summary.total_certificates, 2);
        assert_eq!(summary.head_sha256, sha2);
        assert!(summary.all_signatures_valid);
        assert!(summary.chain_intact);

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn quarantine_store_persists_across_restarts() -> Result<(), ScrubError> {
        let dir = std::env::temp_dir().join(format!("zervram_quarantine_test_{}", Uuid::new_v4()));
        let q_path = dir.join("quarantine.json");
        let dev = Uuid::new_v4();

        {
            let store = QuarantineStore::open(&q_path);
            assert!(store.is_quarantined(dev)?.is_none());
            store.record_quarantine(QuarantineRecord {
                device_uuid: dev,
                pci_bus_id: "0000:00:04.0".to_string(),
                vendor_model: "Tesla T4".to_string(),
                reason: "residual 0xAA poison bytes".to_string(),
                quarantined_at: Utc::now(),
                failed_cert_id: Some(Uuid::now_v7()),
            })?;
        }

        let reloaded = QuarantineStore::open(&q_path);
        let rec = reloaded
            .is_quarantined(dev)?
            .ok_or_else(|| ScrubError::StorageFailure("missing persisted record".to_string()))?;
        assert_eq!(rec.device_uuid, dev);
        assert!(rec.reason.contains("0xAA"));

        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}