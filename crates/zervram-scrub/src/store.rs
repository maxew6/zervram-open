use std::collections::HashMap;

use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::certificate::{ChainAuditSummary, ZeroizationCertificate, GENESIS_PREV_CERT_SHA256};
use crate::error::ScrubError;

pub type BuildAndSignFn =
    Box<dyn FnOnce(String) -> Result<ZeroizationCertificate, ScrubError> + Send>;

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

        let new_hash = cert.certificate_sha256_hex();
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
            if cert.prev_cert_sha256 != expected_prev {
                chain_intact = false;
            }
            if cert.verify_signature(&verifying_key).is_err() {
                all_signatures_valid = false;
            }
            expected_prev = cert.certificate_sha256_hex();
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
