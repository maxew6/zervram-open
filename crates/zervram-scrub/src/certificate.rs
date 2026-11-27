use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zervram_hal::{BackendKind, VerificationMode};

use crate::error::ScrubError;

pub const CERTIFICATE_SCHEMA_V2: &str = "zervram.scrub.v2";

pub const GENESIS_PREV_CERT_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

/// Computes a privacy-preserving SHA-256 hex digest of the departing tenant identifier.
pub fn hash_tenant_id(tenant_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"zervram:tenant:v1|");
    hasher.update(tenant_id.as_bytes());
    hex::encode(hasher.finalize())
}

fn update_len_prefixed(
    hasher: &mut Sha256,
    field: &'static str,
    bytes: &[u8],
) -> Result<(), ScrubError> {
    let len = u32::try_from(bytes.len()).map_err(|_| ScrubError::InvalidCertificateField {
        field,
        details: "variable-length field exceeds u32::MAX bytes".to_string(),
    })?;
    hasher.update(len.to_be_bytes());
    hasher.update(bytes);
    Ok(())
}

/// Signed v2 Ed25519 + SHA-256 VRAM zeroization attestation covering all operational,
/// hardware, verification-coverage, and replay-protection fields inside the signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZeroizationCertificateV2 {
    pub schema_version: String,
    pub cert_id: Uuid,
    pub device_uuid: Uuid,
    pub device_ordinal: u32,
    pub pci_bus_id: String,
    pub vendor_model: String,
    pub backend_kind: BackendKind,
    pub tenant_prev_hash: String,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub swept_bytes: u64,
    pub chunk_count: u32,
    pub chunk_table_sha256: String,
    pub verification_mode: VerificationMode,
    pub verified_bytes: u64,
    pub sampled_pages: u64,
    pub active_processes_before: u32,
    pub challenge_nonce: Option<String>,
    pub duration_us: u64,
    pub verified_zero: bool,
    pub timestamp_utc: DateTime<Utc>,
    pub prev_cert_sha256: String,
    pub ed25519_signature_hex: String,
}

/// Primary certificate type alias for v0.3.0+.
pub type ZeroizationCertificate = ZeroizationCertificateV2;

impl ZeroizationCertificateV2 {
    /// Deterministic, domain-separated, length-prefixed (`u32::to_be_bytes`) SHA-256 preimage
    /// covering every unsigned field in the certificate. Fails closed on out-of-range timestamps.
    pub fn canonical_signing_digest(&self) -> Result<[u8; 32], ScrubError> {
        let nanos = self.timestamp_utc.timestamp_nanos_opt().ok_or_else(|| {
            ScrubError::InvalidCertificateField {
                field: "timestamp_utc",
                details: format!("timestamp {} out of i64 nanosecond range", self.timestamp_utc),
            }
        })?;

        let mut hasher = Sha256::new();
        hasher.update(b"zervram:vram-zero:cert:v2");
        update_len_prefixed(&mut hasher, "schema_version", self.schema_version.as_bytes())?;
        hasher.update(self.cert_id.as_bytes());
        hasher.update(self.device_uuid.as_bytes());
        hasher.update(self.device_ordinal.to_be_bytes());
        update_len_prefixed(&mut hasher, "pci_bus_id", self.pci_bus_id.as_bytes())?;
        update_len_prefixed(&mut hasher, "vendor_model", self.vendor_model.as_bytes())?;
        update_len_prefixed(
            &mut hasher,
            "backend_kind",
            self.backend_kind.as_str().as_bytes(),
        )?;
        update_len_prefixed(
            &mut hasher,
            "tenant_prev_hash",
            self.tenant_prev_hash.as_bytes(),
        )?;
        hasher.update(self.vram_total_bytes.to_be_bytes());
        hasher.update(self.vram_free_bytes.to_be_bytes());
        hasher.update(self.swept_bytes.to_be_bytes());
        hasher.update(self.chunk_count.to_be_bytes());
        update_len_prefixed(
            &mut hasher,
            "chunk_table_sha256",
            self.chunk_table_sha256.as_bytes(),
        )?;
        update_len_prefixed(
            &mut hasher,
            "verification_mode",
            self.verification_mode.as_str().as_bytes(),
        )?;
        hasher.update(self.verified_bytes.to_be_bytes());
        hasher.update(self.sampled_pages.to_be_bytes());
        hasher.update(self.active_processes_before.to_be_bytes());
        match &self.challenge_nonce {
            Some(nonce) => {
                hasher.update([1u8]);
                update_len_prefixed(&mut hasher, "challenge_nonce", nonce.as_bytes())?;
            }
            None => {
                hasher.update([0u8]);
            }
        }
        hasher.update(self.duration_us.to_be_bytes());
        hasher.update([u8::from(self.verified_zero)]);
        hasher.update(nanos.to_be_bytes());
        update_len_prefixed(
            &mut hasher,
            "prev_cert_sha256",
            self.prev_cert_sha256.as_bytes(),
        )?;
        Ok(hasher.finalize().into())
    }

    /// SHA-256 hash of the complete signed certificate, chained into the next certificate's
    /// `prev_cert_sha256` field.
    pub fn certificate_sha256_hex(&self) -> Result<String, ScrubError> {
        let digest = self.canonical_signing_digest()?;
        let mut hasher = Sha256::new();
        hasher.update(b"zervram:vram-zero:cert-hash:v2");
        hasher.update(digest);
        update_len_prefixed(
            &mut hasher,
            "ed25519_signature_hex",
            self.ed25519_signature_hex.as_bytes(),
        )?;
        Ok(hex::encode(hasher.finalize()))
    }

    /// Verifies `ed25519_signature_hex` strictly (`verify_strict`) against `verifying_key`.
    pub fn verify_signature(&self, verifying_key: &VerifyingKey) -> Result<(), ScrubError> {
        let sig_bytes =
            hex::decode(&self.ed25519_signature_hex).map_err(|err| ScrubError::InvalidHex {
                field: "ed25519_signature_hex",
                details: err.to_string(),
            })?;
        let signature = Signature::from_slice(&sig_bytes).map_err(|_| {
            ScrubError::InvalidSignature {
                cert_id: self.cert_id,
            }
        })?;
        let digest = self.canonical_signing_digest()?;
        verifying_key
            .verify_strict(&digest, &signature)
            .map_err(|_| ScrubError::InvalidSignature {
                cert_id: self.cert_id,
            })
    }
}

/// Exported JSON envelope for `zervram-scrub --json` / `--out-cert`.
/// Contains zero unsigned semantic metadata fields: every attestation attribute lives inside
/// the Ed25519-signed `certificate`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationBundle {
    pub ed25519_public_key_hex: String,
    pub certificate_sha256: String,
    pub certificate: ZeroizationCertificateV2,
}

/// Fail-closed policy controlling offline certificate verification.
#[derive(Debug, Clone, Default)]
pub struct VerifyPolicy {
    /// Pinned trusted Ed25519 verifying keys (from `--pubkey`, `--key-file`, or `--trust-dir`).
    pub trusted_keys: Vec<VerifyingKey>,
    /// Explicitly unsafe opt-in (`--insecure-trust-embedded-key`) to verify against an untrusted
    /// self-embedded key in `AttestationBundle`. Defaults to `false` (fail closed).
    pub allow_insecure_embedded_key: bool,
    /// Optional verifier/scheduler nonce that must match `certificate.challenge_nonce`.
    pub expected_nonce: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedBundleSummary {
    pub cert_id: Uuid,
    pub device_uuid: Uuid,
    pub device_ordinal: u32,
    pub vendor_model: String,
    pub backend_kind: BackendKind,
    pub verification_mode: VerificationMode,
    pub swept_bytes: u64,
    pub verified_bytes: u64,
    pub sampled_pages: u64,
    pub duration_us: u64,
    pub computed_cert_sha256: String,
    pub ed25519_public_key_hex: String,
    pub used_insecure_embedded_key: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateVerificationResult {
    pub cert_id: Uuid,
    pub signature_valid: bool,
    pub chain_link_valid: bool,
    pub verified_zero_flag: bool,
    pub computed_cert_sha256: String,
    pub ed25519_public_key_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainAuditSummary {
    pub total_certificates: usize,
    pub head_sha256: String,
    pub all_signatures_valid: bool,
    pub chain_intact: bool,
}

pub fn parse_verifying_key(hex_str: &str) -> Result<VerifyingKey, ScrubError> {
    let raw = hex::decode(hex_str.trim()).map_err(|err| {
        ScrubError::VerificationFailure(format!("invalid public key hex: {err}"))
    })?;
    let bytes: [u8; 32] = raw.try_into().map_err(|_| {
        ScrubError::VerificationFailure(
            "Ed25519 public key must be exactly 32 bytes (64 hex chars)".to_string(),
        )
    })?;
    VerifyingKey::from_bytes(&bytes)
        .map_err(|err| ScrubError::VerificationFailure(format!("invalid Ed25519 public key: {err}")))
}

/// Verifies an exported `AttestationBundle` (or raw `ZeroizationCertificateV2`) JSON string
/// under `policy`. Fails closed unless a pinned trusted key is supplied (or
/// `allow_insecure_embedded_key` is explicitly enabled).
pub fn verify_bundle_json(
    raw_json: &str,
    policy: &VerifyPolicy,
) -> Result<VerifiedBundleSummary, ScrubError> {
    let (cert, embedded_pubkey, embedded_hash) =
        if let Ok(bundle) = serde_json::from_str::<AttestationBundle>(raw_json) {
            (
                bundle.certificate,
                Some(bundle.ed25519_public_key_hex),
                Some(bundle.certificate_sha256),
            )
        } else {
            let cert: ZeroizationCertificateV2 =
                serde_json::from_str(raw_json).map_err(|err| {
                    ScrubError::VerificationFailure(format!(
                        "failed to parse v2 attestation bundle or certificate JSON: {err}"
                    ))
                })?;
            (cert, None, None)
        };

    if cert.schema_version != CERTIFICATE_SCHEMA_V2 {
        return Err(ScrubError::VerificationFailure(format!(
            "unsupported certificate schema_version '{}'; expected '{CERTIFICATE_SCHEMA_V2}'",
            cert.schema_version
        )));
    }

    let (candidate_keys, used_insecure_embedded_key): (Vec<VerifyingKey>, bool) =
        if !policy.trusted_keys.is_empty() {
            if let Some(ref emb_hex) = embedded_pubkey {
                let emb_vk = parse_verifying_key(emb_hex)?;
                if !policy.trusted_keys.contains(&emb_vk) {
                    return Err(ScrubError::VerificationFailure(format!(
                        "embedded public key {emb_hex} does not match any pinned trusted key"
                    )));
                }
            }
            (policy.trusted_keys.clone(), false)
        } else if policy.allow_insecure_embedded_key {
            let emb_hex = embedded_pubkey.as_deref().ok_or_else(|| {
                ScrubError::VerificationFailure(
                    "certificate JSON does not contain an embedded public key".to_string(),
                )
            })?;
            (vec![parse_verifying_key(emb_hex)?], true)
        } else {
            return Err(ScrubError::VerificationFailure(
                "verification requires a pinned trust anchor: provide --pubkey <hex>, --key-file <path>, or --trust-dir <dir> (or pass --insecure-trust-embedded-key only for local debugging)"
                    .to_string(),
            ));
        };

    let matched_key = candidate_keys
        .iter()
        .find(|vk| cert.verify_signature(vk).is_ok())
        .copied()
        .ok_or(ScrubError::InvalidSignature {
            cert_id: cert.cert_id,
        })?;

    let computed_sha = cert.certificate_sha256_hex()?;
    if let Some(expected_sha) = embedded_hash {
        if !expected_sha.eq_ignore_ascii_case(&computed_sha) {
            return Err(ScrubError::VerificationFailure(format!(
                "certificate_sha256 mismatch: expected {expected_sha}, computed {computed_sha}"
            )));
        }
    }

    if let Some(ref expected_nonce) = policy.expected_nonce {
        if cert.challenge_nonce.as_deref() != Some(expected_nonce.as_str()) {
            return Err(ScrubError::VerificationFailure(format!(
                "challenge_nonce mismatch: expected '{expected_nonce}', found {:?}",
                cert.challenge_nonce
            )));
        }
    }

    if !cert.verified_zero {
        return Err(ScrubError::VerificationFailure(format!(
            "certificate {} signature is valid, but verified_zero=false (residual VRAM fault recorded)",
            cert.cert_id
        )));
    }

    if cert.verified_bytes == 0 || cert.sampled_pages == 0 {
        return Err(ScrubError::VerificationFailure(format!(
            "certificate {} reports vacuous verification (verified_bytes={}, sampled_pages={})",
            cert.cert_id, cert.verified_bytes, cert.sampled_pages
        )));
    }

    Ok(VerifiedBundleSummary {
        cert_id: cert.cert_id,
        device_uuid: cert.device_uuid,
        device_ordinal: cert.device_ordinal,
        vendor_model: cert.vendor_model,
        backend_kind: cert.backend_kind,
        verification_mode: cert.verification_mode,
        swept_bytes: cert.swept_bytes,
        verified_bytes: cert.verified_bytes,
        sampled_pages: cert.sampled_pages,
        duration_us: cert.duration_us,
        computed_cert_sha256: computed_sha,
        ed25519_public_key_hex: hex::encode(matched_key.to_bytes()),
        used_insecure_embedded_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signer::{AttestationSigner, Ed25519Signer};

    fn sample_signed_bundle(signer: &Ed25519Signer) -> Result<AttestationBundle, ScrubError> {
        let mut cert = ZeroizationCertificateV2 {
            schema_version: CERTIFICATE_SCHEMA_V2.to_string(),
            cert_id: Uuid::now_v7(),
            device_uuid: Uuid::new_v4(),
            device_ordinal: 0,
            pci_bus_id: "0000:00:04.0".to_string(),
            vendor_model: "Tesla T4".to_string(),
            backend_kind: BackendKind::MockCpu,
            tenant_prev_hash: hash_tenant_id("tenant-alpha"),
            vram_total_bytes: 16 * 1024 * 1024 * 1024,
            vram_free_bytes: 15 * 1024 * 1024 * 1024,
            swept_bytes: 14 * 1024 * 1024 * 1024,
            chunk_count: 56,
            chunk_table_sha256: "a".repeat(64),
            verification_mode: VerificationMode::MockAllMapped,
            verified_bytes: 524_288,
            sampled_pages: 128,
            active_processes_before: 0,
            challenge_nonce: Some("nonce-xyz-123".to_string()),
            duration_us: 2_100,
            verified_zero: true,
            timestamp_utc: Utc::now(),
            prev_cert_sha256: GENESIS_PREV_CERT_SHA256.to_string(),
            ed25519_signature_hex: String::new(),
        };
        let digest = cert.canonical_signing_digest()?;
        cert.ed25519_signature_hex = hex::encode(signer.sign_digest(&digest)?);
        let certificate_sha256 = cert.certificate_sha256_hex()?;
        Ok(AttestationBundle {
            ed25519_public_key_hex: signer.verifying_key_hex(),
            certificate_sha256,
            certificate: cert,
        })
    }

    #[test]
    fn verify_rejects_missing_pinned_pubkey_by_default() -> Result<(), Box<dyn std::error::Error>> {
        let signer = Ed25519Signer::generate_ephemeral();
        let bundle = sample_signed_bundle(&signer)?;
        let json = serde_json::to_string(&bundle)?;

        // Fail closed when no trusted key is pinned.
        let default_policy = VerifyPolicy::default();
        assert!(verify_bundle_json(&json, &default_policy).is_err());

        // Succeeds only when pinned key matches.
        let pinned_policy = VerifyPolicy {
            trusted_keys: vec![signer.verifying_key()],
            allow_insecure_embedded_key: false,
            expected_nonce: Some("nonce-xyz-123".to_string()),
        };
        let verified = verify_bundle_json(&json, &pinned_policy)?;
        assert_eq!(verified.backend_kind, BackendKind::MockCpu);
        assert!(!verified.used_insecure_embedded_key);

        // Fails if attacker swaps in their own keypair while pinned key is configured.
        let attacker = Ed25519Signer::generate_ephemeral();
        let forged_bundle = sample_signed_bundle(&attacker)?;
        let forged_json = serde_json::to_string(&forged_bundle)?;
        assert!(verify_bundle_json(&forged_json, &pinned_policy).is_err());
        Ok(())
    }

    #[test]
    fn tamper_rejection_covers_backend_kind_and_all_v2_fields(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let signer = Ed25519Signer::generate_ephemeral();
        let bundle = sample_signed_bundle(&signer)?;
        let policy = VerifyPolicy {
            trusted_keys: vec![signer.verifying_key()],
            allow_insecure_embedded_key: false,
            expected_nonce: None,
        };

        // 1. Tamper backend_kind: mock_cpu -> cuda_native
        let mut tampered = bundle.clone();
        tampered.certificate.backend_kind = BackendKind::CudaNative;
        tampered.certificate_sha256 = tampered.certificate.certificate_sha256_hex()?;
        assert!(verify_bundle_json(&serde_json::to_string(&tampered)?, &policy).is_err());

        // 2. Tamper device_ordinal
        let mut tampered_ord = bundle.clone();
        tampered_ord.certificate.device_ordinal = 7;
        tampered_ord.certificate_sha256 = tampered_ord.certificate.certificate_sha256_hex()?;
        assert!(verify_bundle_json(&serde_json::to_string(&tampered_ord)?, &policy).is_err());

        // 3. Tamper verified_bytes
        let mut tampered_vb = bundle.clone();
        tampered_vb.certificate.verified_bytes = tampered_vb.certificate.swept_bytes;
        tampered_vb.certificate_sha256 = tampered_vb.certificate.certificate_sha256_hex()?;
        assert!(verify_bundle_json(&serde_json::to_string(&tampered_vb)?, &policy).is_err());

        // 4. Tamper verification_mode
        let mut tampered_vm = bundle.clone();
        tampered_vm.certificate.verification_mode = VerificationMode::GpuPtxFullReduction;
        tampered_vm.certificate_sha256 = tampered_vm.certificate.certificate_sha256_hex()?;
        assert!(verify_bundle_json(&serde_json::to_string(&tampered_vm)?, &policy).is_err());

        // 5. Tamper challenge_nonce
        let mut tampered_nonce = bundle.clone();
        tampered_nonce.certificate.challenge_nonce = Some("forged-nonce".to_string());
        tampered_nonce.certificate_sha256 = tampered_nonce.certificate.certificate_sha256_hex()?;
        assert!(verify_bundle_json(&serde_json::to_string(&tampered_nonce)?, &policy).is_err());

        Ok(())
    }

    #[test]
    fn length_prefixed_canonical_digest_prevents_field_boundary_collisions(
    ) -> Result<(), ScrubError> {
        let signer = Ed25519Signer::generate_ephemeral();
        let mut a = sample_signed_bundle(&signer)?.certificate;
        let mut b = a.clone();
        a.pci_bus_id = "AB".to_string();
        a.vendor_model = "C".to_string();
        b.pci_bus_id = "A".to_string();
        b.vendor_model = "BC".to_string();
        assert_ne!(a.canonical_signing_digest()?, b.canonical_signing_digest()?);
        Ok(())
    }
}