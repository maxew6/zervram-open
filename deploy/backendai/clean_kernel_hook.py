"""Async GPU VRAM sanitization hook for Lablup Backend.AI AbstractAgent.clean_kernel.

Drop-in integration point: `src/ai/backend/agent/docker/agent.py` inside
`DockerAgent.clean_kernel(...)` immediately after container teardown and BEFORE
`alloc_map.py` releases the allocated GPU slots back to the schedulable pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

log = logging.getLogger(__name__)

EXIT_CODE_QUARANTINED = 42


class ScrubQuarantineError(RuntimeError):
    """Raised when Zervram quarantines a GPU (exit code 42 / HTTP 423 Locked)."""

    def __init__(self, device_ordinal: int, detail: str) -> None:
        super().__init__(f"GPU {device_ordinal} quarantined by zervram-scrub: {detail}")
        self.device_ordinal = device_ordinal
        self.detail = detail


class ScrubExecutionError(RuntimeError):
    """Raised when zervram-scrub fails for a non-quarantine operational reason."""

    def __init__(self, device_ordinal: int, exit_code: int, detail: str) -> None:
        super().__init__(f"GPU {device_ordinal} scrub failed (rc={exit_code}): {detail}")
        self.device_ordinal = device_ordinal
        self.exit_code = exit_code
        self.detail = detail


@dataclass(frozen=True)
class ScrubResult:
    device_ordinal: int
    cert_id: str
    certificate_sha256: str
    verification_mode: str
    verified_bytes: int
    swept_bytes: int
    duration_us: int
    raw_bundle: dict


async def scrub_kernel_devices_async(
    device_ordinals: Iterable[int],
    *,
    tenant_id: str,
    kernel_id: str,
    challenge_nonce: Optional[str] = None,
    zervram_bin: str = "/usr/local/bin/zervram-scrub",
    daemon_url: str = "http://127.0.0.1:9740",
    token_file: Optional[str] = "/etc/zervram/daemon.token",
    key_file: str = "/etc/zervram/node.key",
    chain_file: str = "/var/lib/zervram/chain.jsonl",
    timeout_secs: float = 60.0,
) -> list[ScrubResult]:
    """Sanitize each released GPU ordinal via warm zervramd (with --fallback-direct).

    Never uses shell=True. Fails closed on quarantine (exit 42) so the agent can
    withhold the faulted GPU from alloc_map.py and alert the cluster manager.
    """
    results: list[ScrubResult] = []
    for raw_ord in device_ordinals:
        ordinal = int(raw_ord)
        if ordinal < 0:
            raise ValueError(f"invalid negative GPU ordinal: {ordinal}")

        args = [
            zervram_bin,
            "--device",
            str(ordinal),
            "--daemon-url",
            daemon_url,
            "--fallback-direct",
            "--key-file",
            key_file,
            "--chain-file",
            chain_file,
            "--tenant-id",
            tenant_id,
            "--job-id",
            kernel_id,
            "--json",
        ]
        if token_file:
            args.extend(["--token-file", token_file])
        nonce = challenge_nonce or f"bai-{kernel_id}-gpu-{ordinal}"
        args.extend(["--nonce", nonce])

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_secs
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise ScrubExecutionError(ordinal, -1, f"timed out after {timeout_secs}s") from exc

        rc = proc.returncode if proc.returncode is not None else -1
        stderr_txt = stderr_b.decode("utf-8", errors="replace").strip()
        if rc == EXIT_CODE_QUARANTINED:
            log.error("Zervram quarantined GPU %d after kernel %s: %s", ordinal, kernel_id, stderr_txt)
            raise ScrubQuarantineError(ordinal, stderr_txt)
        if rc != 0:
            log.error("Zervram scrub failed on GPU %d (rc=%d): %s", ordinal, rc, stderr_txt)
            raise ScrubExecutionError(ordinal, rc, stderr_txt)

        bundle = json.loads(stdout_b.decode("utf-8"))
        cert = bundle["certificate"]
        results.append(
            ScrubResult(
                device_ordinal=ordinal,
                cert_id=cert["cert_id"],
                certificate_sha256=bundle["certificate_sha256"],
                verification_mode=cert["verification_mode"],
                verified_bytes=int(cert["verified_bytes"]),
                swept_bytes=int(cert["swept_bytes"]),
                duration_us=int(cert["duration_us"]),
                raw_bundle=bundle,
            )
        )
    return results
