#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::Command;

use uuid::Uuid;
use zervram_scrub::{AttestationBundle, GENESIS_PREV_CERT_SHA256};

struct TempDirGuard {
    path: PathBuf,
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn cli_two_run_chain_fail_closed_verify_and_tamper_rejection(
) -> Result<(), Box<dyn std::error::Error>> {
    let bin = env!("CARGO_BIN_EXE_zervram-scrub");
    let tmp_dir = std::env::temp_dir().join(format!("zervram_cli_e2e_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir)?;
    let _guard = TempDirGuard {
        path: tmp_dir.clone(),
    };

    let key_file = tmp_dir.join("node.key");
    let chain_file = tmp_dir.join("chain.jsonl");
    let cert1_path = tmp_dir.join("cert1.json");
    let cert2_path = tmp_dir.join("cert2.json");

    // 1. First CLI invocation (creates node.key + genesis entry in chain.jsonl)
    let out1 = Command::new(bin)
        .args([
            "--allow-mock",
            "--device",
            "0",
            "--key-file",
            &key_file.to_string_lossy(),
            "--chain-file",
            &chain_file.to_string_lossy(),
            "--nonce",
            "cli-nonce-001",
            "--out-cert",
            &cert1_path.to_string_lossy(),
            "--json",
        ])
        .output()?;
    assert!(
        out1.status.success(),
        "run 1 failed: {}",
        String::from_utf8_lossy(&out1.stderr)
    );

    // 2. Second CLI invocation (must link prev_cert_sha256 == cert1.certificate_sha256)
    let out2 = Command::new(bin)
        .args([
            "--allow-mock",
            "--device",
            "0",
            "--key-file",
            &key_file.to_string_lossy(),
            "--chain-file",
            &chain_file.to_string_lossy(),
            "--out-cert",
            &cert2_path.to_string_lossy(),
            "--json",
        ])
        .output()?;
    assert!(
        out2.status.success(),
        "run 2 failed: {}",
        String::from_utf8_lossy(&out2.stderr)
    );

    let bundle1: AttestationBundle = serde_json::from_str(&std::fs::read_to_string(&cert1_path)?)?;
    let bundle2: AttestationBundle = serde_json::from_str(&std::fs::read_to_string(&cert2_path)?)?;

    assert_eq!(bundle1.certificate.prev_cert_sha256, GENESIS_PREV_CERT_SHA256);
    assert_eq!(
        bundle2.certificate.prev_cert_sha256,
        bundle1.certificate_sha256
    );
    assert_eq!(
        bundle1.certificate.challenge_nonce.as_deref(),
        Some("cli-nonce-001")
    );
    let pubkey_hex = bundle1.ed25519_public_key_hex.clone();

    // 3. `verify` WITHOUT --pubkey / --key-file MUST FAIL CLOSED
    let unpinned = Command::new(bin)
        .args(["verify", &cert1_path.to_string_lossy()])
        .output()?;
    assert!(
        !unpinned.status.success(),
        "verify without pinned key should have failed closed"
    );

    // 4. `verify` WITH --pubkey + matching --expected-nonce PASSES
    let pinned_ok = Command::new(bin)
        .args([
            "verify",
            &cert1_path.to_string_lossy(),
            "--pubkey",
            &pubkey_hex,
            "--expected-nonce",
            "cli-nonce-001",
        ])
        .output()?;
    assert!(
        pinned_ok.status.success(),
        "verify with pinned pubkey failed: {}",
        String::from_utf8_lossy(&pinned_ok.stderr)
    );

    // 5. `verify` WITH wrong --expected-nonce FAILS
    let wrong_nonce = Command::new(bin)
        .args([
            "verify",
            &cert1_path.to_string_lossy(),
            "--pubkey",
            &pubkey_hex,
            "--expected-nonce",
            "wrong-nonce",
        ])
        .output()?;
    assert!(!wrong_nonce.status.success());

    // 6. `verify` WITH --insecure-trust-embedded-key succeeds with warning
    let insecure_ok = Command::new(bin)
        .args([
            "verify",
            &cert1_path.to_string_lossy(),
            "--insecure-trust-embedded-key",
        ])
        .output()?;
    assert!(insecure_ok.status.success());

    // 7. Mutating `backend_kind` ("mock_cpu" -> "cuda_native") MUST FAIL verification
    let raw1 = std::fs::read_to_string(&cert1_path)?;
    let tampered_raw = raw1.replacen("\"mock_cpu\"", "\"cuda_native\"", 1);
    assert_ne!(raw1, tampered_raw);
    let tampered_path = tmp_dir.join("cert1_tampered.json");
    std::fs::write(&tampered_path, tampered_raw.as_bytes())?;

    let tampered_check = Command::new(bin)
        .args([
            "verify",
            &tampered_path.to_string_lossy(),
            "--pubkey",
            &pubkey_hex,
        ])
        .output()?;
    assert!(
        !tampered_check.status.success(),
        "tampering backend_kind must fail verification"
    );

    // 8. `verify-chain` WITHOUT --pubkey MUST FAIL CLOSED, WITH --pubkey PASSES
    let chain_unpinned = Command::new(bin)
        .args(["verify-chain", "--chain-file", &chain_file.to_string_lossy()])
        .output()?;
    assert!(!chain_unpinned.status.success());

    let chain_pinned = Command::new(bin)
        .args([
            "verify-chain",
            "--chain-file",
            &chain_file.to_string_lossy(),
            "--pubkey",
            &pubkey_hex,
        ])
        .output()?;
    assert!(
        chain_pinned.status.success(),
        "verify-chain with pinned pubkey failed: {}",
        String::from_utf8_lossy(&chain_pinned.stderr)
    );

    Ok(())
}