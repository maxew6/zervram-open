use std::fmt;
use std::path::Path;

use ed25519_dalek::pkcs8::{DecodePrivateKey, EncodePrivateKey, spki::der::pem::LineEnding};

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};

use crate::error::ScrubError;

pub trait AttestationSigner: Send + Sync + 'static {
    fn sign_digest(&self, digest: &[u8; 32]) -> Result<[u8; 64], ScrubError>;
    fn verify_digest(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), ScrubError>;
    fn verifying_key_bytes(&self) -> [u8; 32];
    fn verifying_key_hex(&self) -> String;
}

/// Ed25519 signer backed by `ed25519-dalek` (`SigningKey` is zeroized on drop).
pub struct Ed25519Signer {
    signing_key: SigningKey,
}

impl fmt::Debug for Ed25519Signer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519Signer")
            .field("verifying_key_hex", &self.verifying_key_hex())
            .finish()
    }
}

impl Ed25519Signer {
    pub fn from_bytes(secret_key: [u8; 32]) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&secret_key),
        }
    }

    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        Self { signing_key }
    }

    pub fn generate_ephemeral() -> Self {
        let mut csprng = rand::rngs::OsRng;
        Self {
            signing_key: SigningKey::generate(&mut csprng),
        }
    }

    pub fn from_pkcs8_pem(pem: &str) -> Result<Self, ScrubError> {
        let signing_key = SigningKey::from_pkcs8_pem(pem)
            .map_err(|err| ScrubError::SigningFailure(format!("invalid PKCS#8 PEM key: {err}")))?;
        Ok(Self { signing_key })
    }

    pub fn to_pkcs8_pem(&self) -> Result<String, ScrubError> {
        let pem = self
            .signing_key
            .to_pkcs8_pem(LineEnding::LF)
            .map_err(|err| ScrubError::SigningFailure(format!("PKCS#8 PEM encoding failed: {err}")))?;
        Ok(pem.as_str().to_string())
    }

    /// Loads an existing PKCS#8 Ed25519 PEM key from `path`, or generates and persists a new key
    /// so minted certificates remain cryptographically verifiable across subprocess runs and reboots.
    /// Returns `(signer, created_new)`.
    pub fn load_or_generate_key_file(path: &Path) -> Result<(Self, bool), ScrubError> {
        if path.exists() {
            let pem = std::fs::read_to_string(path).map_err(|err| {
                ScrubError::SigningFailure(format!(
                    "failed to read signing key '{}': {err}",
                    path.display()
                ))
            })?;
            let signer = Self::from_pkcs8_pem(&pem)?;
            return Ok((signer, false));
        }

        let signer = Self::generate_ephemeral();
        let pem = signer.to_pkcs8_pem()?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    ScrubError::SigningFailure(format!(
                        "failed to create key directory '{}': {err}",
                        parent.display()
                    ))
                })?;
            }
        }

        std::fs::write(path, pem.as_bytes()).map_err(|err| {
            ScrubError::SigningFailure(format!(
                "failed to persist signing key '{}': {err}",
                path.display()
            ))
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(path, perms);
        }

        Ok((signer, true))
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    pub fn clone_signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

impl AttestationSigner for Ed25519Signer {
    fn sign_digest(&self, digest: &[u8; 32]) -> Result<[u8; 64], ScrubError> {
        Ok(self.signing_key.sign(digest).to_bytes())
    }

    fn verify_digest(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), ScrubError> {
        let sig = Signature::from_bytes(signature);
        self.signing_key
            .verifying_key()
            .verify_strict(digest, &sig)
            .map_err(|err| ScrubError::SigningFailure(err.to_string()))
    }

    fn verifying_key_bytes(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    fn verifying_key_hex(&self) -> String {
        hex::encode(self.verifying_key_bytes())
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    #[test]
    fn persists_and_reloads_pkcs8_ed25519_key_across_restarts() -> Result<(), ScrubError> {
        let tmp = std::env::temp_dir().join(format!("zervram_key_{}.key", Uuid::new_v4()));
        let (first, created_first) = Ed25519Signer::load_or_generate_key_file(&tmp)?;
        assert!(created_first);

        let (second, created_second) = Ed25519Signer::load_or_generate_key_file(&tmp)?;
        assert!(!created_second);
        assert_eq!(first.verifying_key_hex(), second.verifying_key_hex());

        let digest = [0x42u8; 32];
        let sig = first.sign_digest(&digest)?;
        second.verify_digest(&digest, &sig)?;

        let _ = std::fs::remove_file(&tmp);
        Ok(())
    }
}