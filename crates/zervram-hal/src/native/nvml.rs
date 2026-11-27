#![allow(unsafe_code)]

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use libloading::Library;
use uuid::Uuid;

use crate::error::HalError;

const NVML_BACKEND_NAME: &str = "nvidia-nvml";
const NVML_SUCCESS: u32 = 0;
const NVML_ERROR_NOT_SUPPORTED: u32 = 3;
const NVML_ERROR_INSUFFICIENT_SIZE: u32 = 7;

#[cfg(target_os = "windows")]
const NVML_DRIVER_CANDIDATES: &[&str] = &[
    "nvml.dll",
    "C:\\Windows\\System32\\nvml.dll",
    "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
];

#[cfg(not(target_os = "windows"))]
const NVML_DRIVER_CANDIDATES: &[&str] = &[
    "libnvidia-ml.so.1",
    "libnvidia-ml.so",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/local/nvidia/lib64/libnvidia-ml.so.1",
];

type NvmlReturn = u32;
type NvmlDevice = *mut c_void;

/// Exact C ABI layout of `nvmlProcessInfo_t` used by `nvmlDeviceGetComputeRunningProcesses_v3`
/// and `nvmlDeviceGetComputeRunningProcesses_v2` (24 bytes on 64-bit platforms).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct NvmlProcessInfo {
    pid: u32,
    used_gpu_memory: u64,
    gpu_instance_id: u32,
    compute_instance_id: u32,
}

type FnNvmlInitV2 = unsafe extern "system" fn() -> NvmlReturn;
type FnNvmlShutdown = unsafe extern "system" fn() -> NvmlReturn;
type FnNvmlDeviceGetHandleByUuid =
    unsafe extern "system" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn;
type FnNvmlDeviceGetHandleByPciBusIdV2 =
    unsafe extern "system" fn(*const c_char, *mut NvmlDevice) -> NvmlReturn;
type FnNvmlDeviceGetHandleByIndexV2 =
    unsafe extern "system" fn(u32, *mut NvmlDevice) -> NvmlReturn;
type FnNvmlDeviceGetComputeRunningProcesses =
    unsafe extern "system" fn(NvmlDevice, *mut u32, *mut NvmlProcessInfo) -> NvmlReturn;
type FnNvmlDeviceGetRemappedRows =
    unsafe extern "system" fn(NvmlDevice, *mut u32, *mut u32, *mut u32, *mut u32) -> NvmlReturn;

#[derive(Debug, Clone)]
pub(crate) struct NvmlPreScrubReport {
    pub(crate) active_other_pids: Vec<u32>,
    pub(crate) ecc_remapping_failure: bool,
}

struct NvmlSession {
    nvml_shutdown: FnNvmlShutdown,
    get_handle_by_uuid: FnNvmlDeviceGetHandleByUuid,
    get_handle_by_pci: Option<FnNvmlDeviceGetHandleByPciBusIdV2>,
    get_handle_by_index: Option<FnNvmlDeviceGetHandleByIndexV2>,
    get_compute_procs: FnNvmlDeviceGetComputeRunningProcesses,
    get_remapped_rows: Option<FnNvmlDeviceGetRemappedRows>,
    // Field drop runs AFTER `Drop::drop(&mut self)`, keeping symbols valid during `nvmlShutdown`.
    _lib: Library,
}

impl Drop for NvmlSession {
    fn drop(&mut self) {
        // SAFETY: `nvmlInit_v2` succeeded during `load`, and `_lib` is still alive during `drop`.
        let rc = unsafe { (self.nvml_shutdown)() };
        if rc != NVML_SUCCESS {
            tracing::warn!(nvml_return = rc, "nvmlShutdown returned non-zero status");
        }
    }
}

fn load_nvml_symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> Result<T, HalError> {
    // SAFETY: Caller supplies the exact NVML C ABI signature for `name`.
    let sym = unsafe { lib.get::<T>(name) }.map_err(|err| HalError::NativeDriverFailure {
        backend: NVML_BACKEND_NAME,
        message: format!(
            "failed to resolve NVML symbol `{}`: {err}",
            String::from_utf8_lossy(name).trim_end_matches('\0')
        ),
    })?;
    Ok(*sym)
}

impl NvmlSession {
    fn load(explicit_path: Option<&str>) -> Result<Self, HalError> {
        let mut attempts: Vec<String> = Vec::new();
        let mut loaded: Option<Library> = None;

        if let Some(path) = explicit_path {
            // SAFETY: Dynamically loading configured NVML library path at runtime.
            match unsafe { Library::new(path) } {
                Ok(lib) => loaded = Some(lib),
                Err(err) => attempts.push(format!("{path}: {err}")),
            }
        } else {
            for &candidate in NVML_DRIVER_CANDIDATES {
                // SAFETY: Dynamically probing standard OS NVML library paths.
                match unsafe { Library::new(candidate) } {
                    Ok(lib) => {
                        loaded = Some(lib);
                        break;
                    }
                    Err(err) => attempts.push(format!("{candidate}: {err}")),
                }
            }
        }

        let Some(lib) = loaded else {
            return Err(HalError::NativeDriverFailure {
                backend: NVML_BACKEND_NAME,
                message: format!(
                    "could not dynamically load NVIDIA Management Library ({})",
                    attempts.join("; ")
                ),
            });
        };

        let nvml_init_v2: FnNvmlInitV2 = load_nvml_symbol(&lib, b"nvmlInit_v2\0")?;
        let nvml_shutdown: FnNvmlShutdown = load_nvml_symbol(&lib, b"nvmlShutdown\0")?;
        let get_handle_by_uuid: FnNvmlDeviceGetHandleByUuid =
            load_nvml_symbol(&lib, b"nvmlDeviceGetHandleByUUID\0")?;
        let get_handle_by_pci: Option<FnNvmlDeviceGetHandleByPciBusIdV2> =
            load_nvml_symbol(&lib, b"nvmlDeviceGetHandleByPciBusId_v2\0").ok();
        let get_handle_by_index: Option<FnNvmlDeviceGetHandleByIndexV2> =
            load_nvml_symbol(&lib, b"nvmlDeviceGetHandleByIndex_v2\0").ok();

        let get_compute_procs: FnNvmlDeviceGetComputeRunningProcesses =
            load_nvml_symbol(&lib, b"nvmlDeviceGetComputeRunningProcesses_v3\0")
                .or_else(|_| load_nvml_symbol(&lib, b"nvmlDeviceGetComputeRunningProcesses_v2\0"))?;

        let get_remapped_rows: Option<FnNvmlDeviceGetRemappedRows> =
            load_nvml_symbol(&lib, b"nvmlDeviceGetRemappedRows\0").ok();

        // SAFETY: `nvmlInit_v2` initializes an NVML reference-counted session.
        let rc = unsafe { (nvml_init_v2)() };
        if rc != NVML_SUCCESS {
            return Err(HalError::NativeDriverFailure {
                backend: NVML_BACKEND_NAME,
                message: format!("nvmlInit_v2 returned nvmlReturn_t code {rc}"),
            });
        }

        Ok(Self {
            nvml_shutdown,
            get_handle_by_uuid,
            get_handle_by_pci,
            get_handle_by_index,
            get_compute_procs,
            get_remapped_rows,
            _lib: lib,
        })
    }

    fn resolve_device_handle(
        &self,
        device_uuid: Uuid,
        pci_bus_id: &str,
        ordinal: u32,
    ) -> Result<NvmlDevice, HalError> {
        let mut handle: NvmlDevice = ptr::null_mut();

        let uuid_str = format!("GPU-{device_uuid}");
        if let Ok(uuid_cstr) = CString::new(uuid_str) {
            // SAFETY: Valid NUL-terminated UUID string and writable output pointer.
            let rc = unsafe { (self.get_handle_by_uuid)(uuid_cstr.as_ptr(), &mut handle) };
            if rc == NVML_SUCCESS && !handle.is_null() {
                return Ok(handle);
            }
        }

        if let Some(get_by_pci) = self.get_handle_by_pci {
            if let Ok(pci_cstr) = CString::new(pci_bus_id) {
                // SAFETY: Valid NUL-terminated PCI bus ID and writable output pointer.
                let rc = unsafe { (get_by_pci)(pci_cstr.as_ptr(), &mut handle) };
                if rc == NVML_SUCCESS && !handle.is_null() {
                    return Ok(handle);
                }
            }
        }

        if let Some(get_by_idx) = self.get_handle_by_index {
            // SAFETY: Valid device ordinal and writable output pointer.
            let rc = unsafe { (get_by_idx)(ordinal, &mut handle) };
            if rc == NVML_SUCCESS && !handle.is_null() {
                return Ok(handle);
            }
        }

        Err(HalError::NativeDriverFailure {
            backend: NVML_BACKEND_NAME,
            message: format!(
                "unable to resolve NVML handle for device {device_uuid} (pci={pci_bus_id}, ordinal={ordinal})"
            ),
        })
    }

    fn query_active_other_pids(&self, handle: NvmlDevice) -> Result<Vec<u32>, HalError> {
        let mut count: u32 = 0;
        // SAFETY: Passing `count = 0` and NULL buffer queries the required process array size.
        let rc = unsafe { (self.get_compute_procs)(handle, &mut count, ptr::null_mut()) };
        if rc == NVML_SUCCESS {
            return Ok(Vec::new());
        }
        if rc != NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(HalError::NativeDriverFailure {
                backend: NVML_BACKEND_NAME,
                message: format!(
                    "nvmlDeviceGetComputeRunningProcesses returned nvmlReturn_t code {rc}"
                ),
            });
        }

        let alloc_len = usize::try_from(count).unwrap_or(0).saturating_add(8);
        let mut infos = vec![NvmlProcessInfo::default(); alloc_len];
        let mut cap = u32::try_from(infos.len()).unwrap_or(u32::MAX);

        // SAFETY: `infos` has `cap` writable `NvmlProcessInfo` elements.
        let rc2 = unsafe { (self.get_compute_procs)(handle, &mut cap, infos.as_mut_ptr()) };
        if rc2 != NVML_SUCCESS {
            return Err(HalError::NativeDriverFailure {
                backend: NVML_BACKEND_NAME,
                message: format!(
                    "nvmlDeviceGetComputeRunningProcesses (read) returned nvmlReturn_t code {rc2}"
                ),
            });
        }

        let reported = usize::try_from(cap).unwrap_or(0).min(infos.len());
        let self_pid = std::process::id();
        let other_pids: Vec<u32> = infos
            .iter()
            .take(reported)
            .map(|info| info.pid)
            .filter(|&pid| pid != 0 && pid != self_pid)
            .collect();

        Ok(other_pids)
    }

    fn query_ecc_remapping_failure(&self, handle: NvmlDevice) -> bool {
        let Some(get_remapped) = self.get_remapped_rows else {
            return false;
        };
        let mut corr_rows: u32 = 0;
        let mut unc_rows: u32 = 0;
        let mut is_pending: u32 = 0;
        let mut failure_occurred: u32 = 0;
        // SAFETY: All four output pointers are valid mutable `u32` stack locations.
        let rc = unsafe {
            (get_remapped)(
                handle,
                &mut corr_rows,
                &mut unc_rows,
                &mut is_pending,
                &mut failure_occurred,
            )
        };
        if rc == NVML_SUCCESS {
            return failure_occurred != 0;
        }
        if rc != NVML_ERROR_NOT_SUPPORTED {
            tracing::debug!(
                nvml_return = rc,
                "nvmlDeviceGetRemappedRows unavailable on this device"
            );
        }
        false
    }
}

/// Probes NVML before CUDA scrubbing to verify 0 other compute processes are attached and no
/// fatal ECC row-remapping failure has occurred.
pub(crate) fn inspect_device_before_scrub(
    explicit_path: Option<&str>,
    device_uuid: Uuid,
    pci_bus_id: &str,
    ordinal: u32,
) -> Result<NvmlPreScrubReport, HalError> {
    let session = NvmlSession::load(explicit_path)?;
    let handle = session.resolve_device_handle(device_uuid, pci_bus_id, ordinal)?;
    let active_other_pids = session.query_active_other_pids(handle)?;
    let ecc_remapping_failure = session.query_ecc_remapping_failure(handle);
    Ok(NvmlPreScrubReport {
        active_other_pids,
        ecc_remapping_failure,
    })
}