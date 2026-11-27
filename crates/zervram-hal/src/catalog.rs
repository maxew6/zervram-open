use crate::error::HalError;
use crate::model::{AcceleratorModel, AcceleratorVendor, HardwareSpec, VRAM_PAGE_SIZE_BYTES};

/// Reference aggregate Total Processing Performance (TPP) ceiling under U.S. BIS Interim Final
/// Rule 91 FR 74970 / 15 CFR Â§ 744.24 (effective November 19, 2026).
///
/// Note: 15 CFR Â§ 744.24 and Â§ 743.10 govern remote-access compute ceilings and reporting; they do
/// **not** mandate GPU VRAM zeroization or Ed25519 scrub certificates. `zervram-open` provides
/// multi-tenant memory sanitization and tamper-evident audit trails for SOC 2, ISO 27001, HIPAA,
/// and defense-in-depth security controls.
pub const AGGREGATE_TPP_STATUTORY_CEILING: u64 = 12_000_000;

/// Reference formula (`15 CFR EAR Part 744`): `TPP = 2 * Mac_TOPS * bit_length`.
pub fn calculate_tpp(mac_tops: f64, bit_length: u16) -> Result<u64, HalError> {
    if !mac_tops.is_finite() || mac_tops <= 0.0 || bit_length == 0 {
        return Err(HalError::InvalidTppParameters {
            mac_tops,
            bit_length,
        });
    }
    let raw = 2.0 * mac_tops * f64::from(bit_length);
    if !raw.is_finite() || raw < 0.0 || raw >= (u64::MAX as f64) {
        return Err(HalError::ArithmeticOverflow {
            context: "calculate_tpp",
        });
    }
    Ok(raw.round() as u64)
}

pub fn hardware_spec(model: AcceleratorModel) -> &'static HardwareSpec {
    match model {
        AcceleratorModel::NvidiaT4 => &SPEC_T4,
        AcceleratorModel::NvidiaL4 => &SPEC_L4,
        AcceleratorModel::NvidiaL40s => &SPEC_L40S,
        AcceleratorModel::NvidiaA100 => &SPEC_A100,
        AcceleratorModel::NvidiaH100Sxm => &SPEC_H100_SXM,
        AcceleratorModel::NvidiaH200Sxm => &SPEC_H200_SXM,
        AcceleratorModel::NvidiaB200Blackwell => &SPEC_B200_BLACKWELL,
        AcceleratorModel::AmdInstinctMi300x => &SPEC_MI300X,
        AcceleratorModel::AmdInstinctMi325x => &SPEC_MI325X,
        AcceleratorModel::AmdInstinctMi350x => &SPEC_MI350X,
        AcceleratorModel::TenstorrentBlackholeP150 => &SPEC_BLACKHOLE_P150,
        AcceleratorModel::FuriosaRngd => &SPEC_FURIOSA_RNGD,
        AcceleratorModel::FuriosaRngdSriovSlice => &SPEC_FURIOSA_RNGD_SLICE,
        AcceleratorModel::Unknown => &SPEC_UNKNOWN,
    }
}

pub fn all_hardware_specs() -> &'static [&'static HardwareSpec] {
    static ALL_SPECS: [&HardwareSpec; 13] = [
        &SPEC_T4,
        &SPEC_L4,
        &SPEC_L40S,
        &SPEC_A100,
        &SPEC_H100_SXM,
        &SPEC_H200_SXM,
        &SPEC_B200_BLACKWELL,
        &SPEC_MI300X,
        &SPEC_MI325X,
        &SPEC_MI350X,
        &SPEC_BLACKHOLE_P150,
        &SPEC_FURIOSA_RNGD,
        &SPEC_FURIOSA_RNGD_SLICE,
    ];
    &ALL_SPECS
}

/// Checked aggregate TPP multiplication for `quantity` units of `model`.
pub fn total_tpp_for_units(model: AcceleratorModel, quantity: u32) -> Result<u64, HalError> {
    let spec = hardware_spec(model);
    let tpp = spec.tpp_per_unit.ok_or_else(|| HalError::UnknownHardwareModel {
        raw: format!("{model:?}"),
    })?;
    tpp.checked_mul(u64::from(quantity))
        .ok_or(HalError::ArithmeticOverflow {
            context: "total_tpp_for_units",
        })
}

pub fn resolve_model_from_k8s_resource(
    resource_key: &str,
    model_hint: Option<&str>,
) -> Result<AcceleratorModel, HalError> {
    match resource_key.trim() {
        "nvidia.com/t4" => Ok(AcceleratorModel::NvidiaT4),
        "nvidia.com/l4" => Ok(AcceleratorModel::NvidiaL4),
        "nvidia.com/l40s" => Ok(AcceleratorModel::NvidiaL40s),
        "nvidia.com/a100" => Ok(AcceleratorModel::NvidiaA100),
        "nvidia.com/h100" | "nvidia.com/h100-sxm" => Ok(AcceleratorModel::NvidiaH100Sxm),
        "nvidia.com/h200" | "nvidia.com/h200-sxm" => Ok(AcceleratorModel::NvidiaH200Sxm),
        "nvidia.com/b200" | "nvidia.com/b200-blackwell" => {
            Ok(AcceleratorModel::NvidiaB200Blackwell)
        }
        "nvidia.com/gpu" => {
            let normalized = model_hint.map(|s| s.trim().to_ascii_lowercase());
            match normalized.as_deref() {
                Some("t4") | Some("tesla-t4") | Some("nvidia-t4") => Ok(AcceleratorModel::NvidiaT4),
                Some("l4") | Some("nvidia-l4") => Ok(AcceleratorModel::NvidiaL4),
                Some("l40s") | Some("nvidia-l40s") => Ok(AcceleratorModel::NvidiaL40s),
                Some("a100") | Some("nvidia-a100") => Ok(AcceleratorModel::NvidiaA100),
                Some("h100") | Some("nvidia-h100") | Some("h100-sxm") => {
                    Ok(AcceleratorModel::NvidiaH100Sxm)
                }
                Some("h200") | Some("nvidia-h200") | Some("h200-sxm") => {
                    Ok(AcceleratorModel::NvidiaH200Sxm)
                }
                Some("b200") | Some("nvidia-b200") | Some("b200-blackwell") => {
                    Ok(AcceleratorModel::NvidiaB200Blackwell)
                }
                None => Ok(AcceleratorModel::Unknown),
                Some(other) => Err(HalError::UnknownHardwareModel {
                    raw: other.to_string(),
                }),
            }
        }
        "amd.com/mi300x" => Ok(AcceleratorModel::AmdInstinctMi300x),
        "amd.com/mi325x" => Ok(AcceleratorModel::AmdInstinctMi325x),
        "amd.com/mi350x" => Ok(AcceleratorModel::AmdInstinctMi350x),
        "amd.com/gpu" => Ok(AcceleratorModel::Unknown),
        "tenstorrent.com/blackhole-p150" | "tenstorrent.com/blackhole" => {
            Ok(AcceleratorModel::TenstorrentBlackholeP150)
        }
        "furiosa.ai/rngd" => Ok(AcceleratorModel::FuriosaRngd),
        "furiosa.ai/rngd-sriov" | "furiosa.ai/rngd-slice" => {
            Ok(AcceleratorModel::FuriosaRngdSriovSlice)
        }
        other => Err(HalError::UnknownHardwareModel {
            raw: other.to_string(),
        }),
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

pub const SPEC_T4: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaT4,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA Tesla T4",
    tpp_per_unit: Some(2_080),
    mac_tops: 130.0,
    bit_length: 8,
    vram_total_bytes: 16 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(5_769),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/t4", "nvidia.com/gpu"],
};

pub const SPEC_L4: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaL4,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA L4",
    tpp_per_unit: Some(3_872),
    mac_tops: 242.0,
    bit_length: 8,
    vram_total_bytes: 24 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(3_099),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/l4", "nvidia.com/gpu"],
};

pub const SPEC_L40S: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaL40s,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA L40S",
    tpp_per_unit: Some(11_728),
    mac_tops: 733.0,
    bit_length: 8,
    vram_total_bytes: 48 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(1_023),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/l40s", "nvidia.com/gpu"],
};

pub const SPEC_A100: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaA100,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA A100",
    tpp_per_unit: Some(9_984),
    mac_tops: 624.0,
    bit_length: 8,
    vram_total_bytes: 80 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(1_201),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/a100", "nvidia.com/gpu"],
};

pub const SPEC_H100_SXM: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaH100Sxm,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA H100 SXM",
    tpp_per_unit: Some(15_832),
    mac_tops: 989.5,
    bit_length: 8,
    vram_total_bytes: 80 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(757),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/h100", "nvidia.com/gpu"],
};

pub const SPEC_H200_SXM: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaH200Sxm,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA H200 SXM",
    tpp_per_unit: Some(15_832),
    mac_tops: 989.5,
    bit_length: 8,
    vram_total_bytes: 141 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(757),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/h200", "nvidia.com/gpu"],
};

pub const SPEC_B200_BLACKWELL: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::NvidiaB200Blackwell,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "NVIDIA B200 Blackwell",
    tpp_per_unit: Some(36_000),
    mac_tops: 2_250.0,
    bit_length: 8,
    vram_total_bytes: 192 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(333),
    sriov_slice_count: 1,
    k8s_resource_keys: &["nvidia.com/b200", "nvidia.com/gpu"],
};

pub const SPEC_MI300X: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::AmdInstinctMi300x,
    vendor: AcceleratorVendor::Amd,
    canonical_name: "AMD Instinct MI300X",
    tpp_per_unit: Some(20_880),
    mac_tops: 1_305.0,
    bit_length: 8,
    vram_total_bytes: 192 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(574),
    sriov_slice_count: 1,
    k8s_resource_keys: &["amd.com/mi300x", "amd.com/gpu"],
};

pub const SPEC_MI325X: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::AmdInstinctMi325x,
    vendor: AcceleratorVendor::Amd,
    canonical_name: "AMD Instinct MI325X",
    tpp_per_unit: Some(20_880),
    mac_tops: 1_305.0,
    bit_length: 8,
    vram_total_bytes: 256 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(574),
    sriov_slice_count: 1,
    k8s_resource_keys: &["amd.com/mi325x", "amd.com/gpu"],
};

pub const SPEC_MI350X: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::AmdInstinctMi350x,
    vendor: AcceleratorVendor::Amd,
    canonical_name: "AMD Instinct MI350X",
    tpp_per_unit: Some(36_800),
    mac_tops: 2_300.0,
    bit_length: 8,
    vram_total_bytes: 288 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(326),
    sriov_slice_count: 1,
    k8s_resource_keys: &["amd.com/mi350x", "amd.com/gpu"],
};

pub const SPEC_BLACKHOLE_P150: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::TenstorrentBlackholeP150,
    vendor: AcceleratorVendor::Tenstorrent,
    canonical_name: "Tenstorrent Blackhole p150",
    tpp_per_unit: Some(9_400),
    mac_tops: 587.5,
    bit_length: 8,
    vram_total_bytes: 32 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(1_276),
    sriov_slice_count: 1,
    k8s_resource_keys: &["tenstorrent.com/blackhole-p150"],
};

pub const SPEC_FURIOSA_RNGD: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::FuriosaRngd,
    vendor: AcceleratorVendor::FuriosaAi,
    canonical_name: "FuriosaAI RNGD",
    tpp_per_unit: Some(8_192),
    mac_tops: 512.0,
    bit_length: 8,
    vram_total_bytes: 48 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(1_464),
    sriov_slice_count: 8,
    k8s_resource_keys: &["furiosa.ai/rngd"],
};

pub const SPEC_FURIOSA_RNGD_SLICE: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::FuriosaRngdSriovSlice,
    vendor: AcceleratorVendor::FuriosaAi,
    canonical_name: "FuriosaAI RNGD (SR-IOV 1/8)",
    tpp_per_unit: Some(1_024),
    mac_tops: 64.0,
    bit_length: 8,
    vram_total_bytes: 6 * GIB,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: Some(11_718),
    sriov_slice_count: 1,
    k8s_resource_keys: &["furiosa.ai/rngd-sriov"],
};

pub const SPEC_UNKNOWN: HardwareSpec = HardwareSpec {
    model: AcceleratorModel::Unknown,
    vendor: AcceleratorVendor::Nvidia,
    canonical_name: "Unknown Accelerator",
    tpp_per_unit: None,
    mac_tops: 0.0,
    bit_length: 0,
    vram_total_bytes: 0,
    page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    max_units_under_12m_tpp: None,
    sriov_slice_count: 1,
    k8s_resource_keys: &[],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_specs_match_formula_and_12m_ceilings() -> Result<(), HalError> {
        for &spec in all_hardware_specs() {
            let tpp = spec.tpp_per_unit.ok_or_else(|| HalError::UnknownHardwareModel {
                raw: spec.canonical_name.to_string(),
            })?;
            let max_units = spec
                .max_units_under_12m_tpp
                .ok_or_else(|| HalError::UnknownHardwareModel {
                    raw: spec.canonical_name.to_string(),
                })?;

            let computed = calculate_tpp(spec.mac_tops, spec.bit_length)?;
            assert_eq!(computed, tpp);

            let max_allowed_tpp = total_tpp_for_units(spec.model, max_units)?;
            assert!(max_allowed_tpp < AGGREGATE_TPP_STATUTORY_CEILING);

            let breach_tpp = total_tpp_for_units(spec.model, max_units + 1)?;
            assert!(breach_tpp >= AGGREGATE_TPP_STATUTORY_CEILING);
        }

        assert_eq!(hardware_spec(AcceleratorModel::Unknown).tpp_per_unit, None);
        assert_eq!(
            hardware_spec(AcceleratorModel::Unknown).max_units_under_12m_tpp,
            None
        );
        assert!(matches!(
            total_tpp_for_units(AcceleratorModel::Unknown, 1),
            Err(HalError::UnknownHardwareModel { .. })
        ));

        assert_eq!(hardware_spec(AcceleratorModel::NvidiaT4).tpp_per_unit, Some(2_080));
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaH100Sxm).tpp_per_unit, Some(15_832));
        assert_eq!(hardware_spec(AcceleratorModel::NvidiaB200Blackwell).tpp_per_unit, Some(36_000));
        assert_eq!(hardware_spec(AcceleratorModel::AmdInstinctMi300x).tpp_per_unit, Some(20_880));
        assert_eq!(hardware_spec(AcceleratorModel::AmdInstinctMi350x).tpp_per_unit, Some(36_800));
        Ok(())
    }

    #[test]
    fn calculate_tpp_rejects_invalid_and_overflow_inputs() {
        assert!(matches!(
            calculate_tpp(0.0, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(-100.0, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(f64::NAN, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(f64::INFINITY, 8),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(500.0, 0),
            Err(HalError::InvalidTppParameters { .. })
        ));
        assert!(matches!(
            calculate_tpp(1e30, 8),
            Err(HalError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn resolve_k8s_resources_for_all_supported_vendors() -> Result<(), HalError> {
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/t4", None)?,
            AcceleratorModel::NvidiaT4
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/h100", None)?,
            AcceleratorModel::NvidiaH100Sxm
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/h200", None)?,
            AcceleratorModel::NvidiaH200Sxm
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/b200", None)?,
            AcceleratorModel::NvidiaB200Blackwell
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", Some("L40S"))?,
            AcceleratorModel::NvidiaL40s
        );
        assert_eq!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", None)?,
            AcceleratorModel::Unknown
        );
        assert_eq!(
            resolve_model_from_k8s_resource("amd.com/mi300x", None)?,
            AcceleratorModel::AmdInstinctMi300x
        );
        assert_eq!(
            resolve_model_from_k8s_resource("amd.com/mi325x", None)?,
            AcceleratorModel::AmdInstinctMi325x
        );
        assert_eq!(
            resolve_model_from_k8s_resource("amd.com/mi350x", None)?,
            AcceleratorModel::AmdInstinctMi350x
        );
        assert_eq!(
            resolve_model_from_k8s_resource("amd.com/gpu", None)?,
            AcceleratorModel::Unknown
        );
        assert!(matches!(
            resolve_model_from_k8s_resource("vendor.invalid/tpu", None),
            Err(HalError::UnknownHardwareModel { .. })
        ));
        assert!(matches!(
            resolve_model_from_k8s_resource("nvidia.com/gpu", Some("unknown-sku")),
            Err(HalError::UnknownHardwareModel { .. })
        ));
        Ok(())
    }
}