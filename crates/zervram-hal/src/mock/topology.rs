use std::collections::BTreeMap;

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

use crate::catalog::hardware_spec;
use crate::mock::vram_pool::SimulatedVramPool;
use crate::mock::{MockClusterConfig, MockDeviceRecord};
use crate::model::{
    AcceleratorModel, DeviceDescriptor, DeviceState, MemoryRange, VRAM_PAGE_SIZE_BYTES,
};

/// Base 4 GiB-aligned simulated device virtual address space (`0x1_0000_0000` = `4,294,967,296`).
pub const MOCK_VRAM_BASE_ADDR: u64 = 0x1_0000_0000;

pub(crate) fn next_deterministic_uuid(rng: &mut ChaCha8Rng) -> Uuid {
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    uuid::Builder::from_random_bytes(bytes).into_uuid()
}

fn next_device_rng(master_rng: &mut ChaCha8Rng) -> ChaCha8Rng {
    let mut seed = [0u8; 32];
    master_rng.fill_bytes(&mut seed);
    ChaCha8Rng::from_seed(seed)
}

fn aligned_range(start_addr: u64, byte_len: u64) -> MemoryRange {
    MemoryRange::new(start_addr, byte_len).unwrap_or(MemoryRange {
        start_addr: MOCK_VRAM_BASE_ADDR,
        byte_len: u64::from(VRAM_PAGE_SIZE_BYTES),
        page_size_bytes: VRAM_PAGE_SIZE_BYTES,
    })
}

fn insert_whole_cards(
    devices: &mut BTreeMap<Uuid, MockDeviceRecord>,
    master_rng: &mut ChaCha8Rng,
    model: AcceleratorModel,
    count: usize,
    node_prefix: &str,
    pci_bus_base: u32,
    backing_pages: usize,
) {
    let spec = hardware_spec(model);
    for idx in 0..count {
        let device_uuid = next_deterministic_uuid(master_rng);
        let node_ordinal = (idx / 8).saturating_add(1);
        let bus_offset = u32::try_from(idx % 200).unwrap_or(0);
        let pci_bus = pci_bus_base.saturating_add(bus_offset);

        let descriptor = DeviceDescriptor {
            device_uuid,
            node_name: format!("{node_prefix}-{node_ordinal:02}"),
            pci_bus_id: format!("0000:{pci_bus:02x}:00.0"),
            vendor: spec.vendor,
            model: spec.model,
            vendor_model_label: spec.canonical_name.to_string(),
            sriov_slice_index: None,
            sriov_parent_uuid: None,
            tpp_rating: spec.tpp_per_unit,
            vram_total_bytes: spec.vram_total_bytes,
            state: DeviceState::Available,
            last_cert_id: None,
            last_scrub_duration_us: None,
        };

        let range = aligned_range(MOCK_VRAM_BASE_ADDR, spec.vram_total_bytes);
        let vram_pool = SimulatedVramPool::new(range, backing_pages);
        let rng = next_device_rng(master_rng);

        devices.insert(
            device_uuid,
            MockDeviceRecord {
                descriptor,
                vram_pool,
                rng,
            },
        );
    }
}

fn insert_furiosa_rngd_sriov_cards(
    devices: &mut BTreeMap<Uuid, MockDeviceRecord>,
    master_rng: &mut ChaCha8Rng,
    physical_cards: usize,
    backing_pages: usize,
) {
    let parent_spec = hardware_spec(AcceleratorModel::FuriosaRngd);
    let slice_spec = hardware_spec(AcceleratorModel::FuriosaRngdSriovSlice);

    for card_idx in 0..physical_cards {
        let parent_uuid = next_deterministic_uuid(master_rng);
        let node_ordinal = (card_idx / 4).saturating_add(1);
        let bus_offset = u32::try_from(card_idx % 140).unwrap_or(0);
        let pci_bus = 0x65u32.saturating_add(bus_offset);

        for slice_idx in 0..parent_spec.sriov_slice_count {
            let device_uuid = next_deterministic_uuid(master_rng);
            let slice_vram_offset =
                u64::from(slice_idx).saturating_mul(slice_spec.vram_total_bytes);
            let start_addr = MOCK_VRAM_BASE_ADDR.saturating_add(slice_vram_offset);

            let descriptor = DeviceDescriptor {
                device_uuid,
                node_name: format!("npu-node-rngd-{node_ordinal:02}"),
                pci_bus_id: format!("0000:{pci_bus:02x}:00.{slice_idx}"),
                vendor: slice_spec.vendor,
                model: slice_spec.model,
                vendor_model_label: slice_spec.canonical_name.to_string(),
                sriov_slice_index: Some(slice_idx),
                sriov_parent_uuid: Some(parent_uuid),
                tpp_rating: slice_spec.tpp_per_unit,
                vram_total_bytes: slice_spec.vram_total_bytes,
                state: DeviceState::Available,
                last_cert_id: None,
                last_scrub_duration_us: None,
            };

            let range = aligned_range(start_addr, slice_spec.vram_total_bytes);
            let vram_pool = SimulatedVramPool::new(range, backing_pages);
            let rng = next_device_rng(master_rng);

            devices.insert(
                device_uuid,
                MockDeviceRecord {
                    descriptor,
                    vram_pool,
                    rng,
                },
            );
        }
    }
}

/// Builds a deterministic heterogeneous cluster spanning all 5 statutory hardware archetypes:
/// NVIDIA H100 SXM, NVIDIA B200 Blackwell, AMD Instinct MI300X, Tenstorrent Blackhole p150,
/// and FuriosaAI RNGD (expanded into 8x 1,024-TPP SR-IOV slices per physical card).
pub(crate) fn build_mock_cluster(config: &MockClusterConfig) -> BTreeMap<Uuid, MockDeviceRecord> {
    let mut master_rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut devices = BTreeMap::new();
    let backing_pages = config.backing_pages_per_device.max(1);

    insert_whole_cards(
        &mut devices,
        &mut master_rng,
        AcceleratorModel::NvidiaH100Sxm,
        config.h100_count,
        "gpu-node-h100",
        0x0a,
        backing_pages,
    );
    insert_whole_cards(
        &mut devices,
        &mut master_rng,
        AcceleratorModel::NvidiaB200Blackwell,
        config.b200_count,
        "gpu-node-b200",
        0x1b,
        backing_pages,
    );
    insert_whole_cards(
        &mut devices,
        &mut master_rng,
        AcceleratorModel::AmdInstinctMi300x,
        config.mi300x_count,
        "gpu-node-mi300x",
        0x3a,
        backing_pages,
    );
    insert_whole_cards(
        &mut devices,
        &mut master_rng,
        AcceleratorModel::TenstorrentBlackholeP150,
        config.blackhole_p150_count,
        "npu-node-blackhole",
        0x4c,
        backing_pages,
    );
    insert_furiosa_rngd_sriov_cards(
        &mut devices,
        &mut master_rng,
        config.furiosa_rngd_physical_cards,
        backing_pages,
    );

    devices
}
