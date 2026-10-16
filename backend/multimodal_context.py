import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_dataset"
CSV_PATH = GOLDEN_DIR / "oscilloscope_telemetry_dump.csv"
SVG_PATH = GOLDEN_DIR / "AXI_Bus_Timing_Schematic.svg"

DEFAULT_SPEC: Dict[str, Any] = {
    "dma_align_bytes": 64,
    "rsvd_mbz_bit": 7,
    "num_banks": 8,
    "sla_target_ms": 3.50,
}


def ensure_oscilloscope_csv(num_rows: int = 2500) -> Path:
    """
    Generates a 2,500-row high-frequency mixed-signal oscilloscope & on-chip telemetry CSV dump
    if it does not already exist or has fewer than num_rows rows.
    """
    if CSV_PATH.exists():
        try:
            with CSV_PATH.open("r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            if line_count >= num_rows:
                return CSV_PATH
        except Exception:
            pass

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_idx",
        "timestamp_us",
        "pc_addr",
        "dma_cr_hex",
        "axi_wvalid",
        "axi_wready",
        "vdd_core_mv",
        "l1_dirty_lines",
        "active_sram_bank",
        "bank_conflict_ways",
        "die_hotspot_temp_c",
        "event_tag",
    ]

    rows: List[Dict[str, Any]] = []
    for idx in range(num_rows):
        t_us = round(idx * 0.5, 2)
        pc = f"0x0800{0x1000 + (idx % 256) * 4:04X}"
        dma_cr = "0x00"
        wvalid = 0
        wready = 1
        vdd_mv = round(850.0 - 2.5 * math.sin(idx * 0.07), 1)
        dirty_lines = 0
        sram_bank = idx % 8
        conflict_ways = 1
        temp_c = round(54.0 + 4.0 * math.sin(idx * 0.01), 1)
        event_tag = "NOMINAL_IDLE"

        if 200 <= idx < 280:
            pc = "0x08001140"
            dirty_lines = min(56, (idx - 200) + 1)
            vdd_mv = 844.2
            temp_c = 58.5
            event_tag = "CPU_TENSOR_INIT_L1_DIRTY"
        elif 280 <= idx <= 340:
            # Stage 1 Fault Signature around t = 142.50 us (idx 285)
            pc = "0x08001208"
            dma_cr = "0xC5"
            wvalid = 1
            wready = 0  # AXI Arbiter Locked!
            vdd_mv = 682.4 if idx == 285 else round(690.0 + (idx - 280) * 0.4, 1)
            dirty_lines = 56
            conflict_ways = 0
            temp_c = round(89.5 + (idx - 280) * 0.12, 1)
            event_tag = (
                "TRAP_AXI_ARB_LOCK_AND_UNALIGNED_ADDR"
                if idx == 285
                else "AXI_WREADY_STUCK_LOW"
            )
        elif 820 <= idx <= 920:
            # Stage 2 Cache Coherency Corruption Signature around t = 418.00 us (idx 836)
            pc = "0x0800124C"
            dma_cr = "0x05"
            wvalid = 1
            wready = 1
            vdd_mv = 815.0
            dirty_lines = 56  # Unflushed dirty cache lines!
            temp_c = 71.2
            event_tag = (
                "WARN_STALE_L1_CACHE_LINE_READ_NO_DMB"
                if idx == 836
                else "DMA_BYPASS_DIRTY_L1_CACHE"
            )
        elif idx >= 1800:
            # Stage 3 SRAM Bank 0 Conflict & 98.4C Thermal Hotspot around t = 1120.50 us (idx 2241)
            pc = "0x080013A0"
            dma_cr = "0x05"
            wvalid = 1
            wready = 1
            sram_bank = 0  # Stuck hitting Bank 0 on every column gather!
            conflict_ways = 8
            vdd_mv = 792.5
            ramp = min(1.0, (idx - 1800) / 441.0)
            temp_c = round(68.0 + 30.4 * ramp, 1)
            if idx == 2241:
                temp_c = 98.4
                event_tag = "THERMAL_THROTTLE_BANK0_HOTSPOT_98C"
            else:
                event_tag = "SRAM_BANK0_8WAY_COLLISION_STALL"

        rows.append(
            {
                "sample_idx": idx,
                "timestamp_us": f"{t_us:.2f}",
                "pc_addr": pc,
                "dma_cr_hex": dma_cr,
                "axi_wvalid": wvalid,
                "axi_wready": wready,
                "vdd_core_mv": f"{vdd_mv:.1f}",
                "l1_dirty_lines": dirty_lines,
                "active_sram_bank": sram_bank,
                "bank_conflict_ways": conflict_ways,
                "die_hotspot_temp_c": f"{temp_c:.1f}",
                "event_tag": event_tag,
            }
        )

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return CSV_PATH


def build_waveform_event_time_json(
    spec_config: Optional[Dict[str, Any]] = None,
    min_vdd_ts: str = "142.50",
    min_vdd_mv: float = 682.4,
    stale_cache_ts: str = "418.00",
    dirty_lines: int = 56,
    peak_temp_ts: str = "1120.50",
    peak_temp_c: float = 98.4,
) -> Dict[str, Any]:
    """
    WaveformQA-Style Event-Time JSON Normalization:
    Converts raw 2,500-row mixed-signal oscilloscope CSV streams into discrete,
    timestamp-synchronized hardware event windows mapped to source PC and schematic nets.
    """
    cfg = dict(DEFAULT_SPEC)
    if spec_config:
        cfg.update(spec_config)
    align_b = int(cfg["dma_align_bytes"])
    rsvd_bit = int(cfg["rsvd_mbz_bit"])
    rsvd_hex = f"0x{1 << rsvd_bit:02X}"
    num_banks = int(cfg["num_banks"])

    events = [
        {
            "event_id": "EV-AXI-DEADLOCK-01",
            "timestamp_us": min_vdd_ts,
            "window_us": "140.00..170.00",
            "duration_cycles": 217,
            "pc_addr": "0x08001208",
            "source_mapping": "kernel.cpp:line_31",
            "normalized_predicate": (
                f"t={min_vdd_ts}us: AXI_AWVALID=1 & AWREADY=0 for 217 cycles | "
                f"PC=kernel.cpp:line_31 (0x08001208) | DMA_DESC_ACTIVE | "
                f"DMA_CR[Bit {rsvd_bit}]=1 ({rsvd_hex}) & ADDR_ALIGN_MOD_{align_b}!=0"
            ),
            "schematic_net": "NET_DMA_CR_MBZ_TRAP & NET_AXI_AWADDR[6:0]",
            "errata_override_id": "#E-401",
            "physical_causal_mechanism": (
                f"PDN Transient di/dt Spike (18.4 A/ns): Unaligned DMA trap + Bit {rsvd_bit} bus lock "
                f"causes rapid AXI4 arbiter retry oscillation across package inductance (L_pkg * di/dt), "
                f"drooping VDD_CORE from 850.0 mV to {min_vdd_mv:.1f} mV."
            ),
            "observed_metrics": {
                "vdd_core_mv": min_vdd_mv,
                "axi_wvalid": 1,
                "axi_wready": 0,
                "stall_cycles": 217,
            },
        },
        {
            "event_id": "EV-L1-SNOOP-HAZARD-02",
            "timestamp_us": stale_cache_ts,
            "window_us": "410.00..460.00",
            "duration_cycles": 128,
            "pc_addr": "0x0800124C",
            "source_mapping": "kernel.cpp:line_35",
            "normalized_predicate": (
                f"t={stale_cache_ts}us: CPU_DMB_FENCE_N=1 (INACTIVE) & L1_DIRTY_LINES={dirty_lines} | "
                f"PC=kernel.cpp:line_35 (0x0800124C) | NON_COHERENT_DMA_READ_ACTIVE"
            ),
            "schematic_net": "NET_DMB_SNOOP_INV",
            "errata_override_id": "#E-402",
            "physical_causal_mechanism": (
                f"Non-Coherent DMA Snoop Bypass: Rev 4.2 snoop filter defect (#E-402) fails to flush "
                f"{dirty_lines} dirty L1 cache lines before DMA burst and compiler -O2 hoists non-volatile "
                f"DMA_SR reads, corrupting 87.5% of tensor words."
            ),
            "observed_metrics": {
                "l1_dirty_lines": dirty_lines,
                "dmb_fence_asserted": False,
                "corrupted_words_ratio": 0.875,
            },
        },
        {
            "event_id": "EV-SRAM-WAVEFRONT-HOTSPOT-03",
            "timestamp_us": peak_temp_ts,
            "window_us": "900.00..1249.50",
            "duration_cycles": 28672,
            "pc_addr": "0x080013A0",
            "source_mapping": "kernel.cpp:line_48",
            "normalized_predicate": (
                f"t={peak_temp_ts}us: SRAM_BANK_SEL=0 & CONFLICT_WAYS={num_banks} for 28,672 cycles | "
                f"PC=kernel.cpp:line_48 (0x080013A0) | STRIDE={num_banks * 4}B_WAVEFRONT_SERIALIZATION"
            ),
            "schematic_net": "NET_SRAM_XBAR_ARB & DIE_BANK0_SENSE",
            "errata_override_id": "#E-403",
            "physical_causal_mechanism": (
                f"Wavefront Serialization & Bank 0 Power Density Concentration: {num_banks * 4}-Byte "
                f"column-gather stride maps all {num_banks} vector lanes to Bank 0 "
                f"(({num_banks}*r + c) mod {num_banks} = c), causing {num_banks}-way serialization stalls "
                f"and concentrating 45.9 W/mm² switching power in Bank 0 sense amplifiers ({peak_temp_c:.1f}°C)."
            ),
            "observed_metrics": {
                "active_sram_bank": 0,
                "bank_conflict_ways": num_banks,
                "die_hotspot_temp_c": peak_temp_c,
                "serialization_penalty_cycles_per_access": num_banks * 12,
            },
        },
    ]

    for ev in events:
        ev["timestamp_window_us"] = ev["window_us"]
        ev["signal_predicate"] = ev["normalized_predicate"]
        ev["pc_symbol"] = f"{ev['source_mapping']} ({ev['pc_addr']})"

    return {
        "schema": "WaveformQA-EventTime-Telemetry/v3.0",
        "clock_domain_core_mhz": 400,
        "clock_domain_axi_mhz": 200,
        "total_raw_csv_samples": 2500,
        "normalized_events": events,
        "events": events,
    }


def build_hardware_spec_ir(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    SpecMap / Qiao & Dick (2026) Typed HardwareSpec IR Builder:
    Synthesizes a formal intermediate representation combining nominal datasheet tables,
    Cross-Page Silicon Errata Overrides (#E-401, #E-402, #E-403), schematic netlists,
    and deterministic C++17 static_assert verification guards.
    """
    cfg = dict(DEFAULT_SPEC)
    if spec_config:
        cfg.update(spec_config)

    align_b = int(cfg["dma_align_bytes"])
    rsvd_bit = int(cfg["rsvd_mbz_bit"])
    rsvd_hex = f"0x{1 << rsvd_bit:02X}"
    num_banks = int(cfg["num_banks"])
    padded_stride = num_banks + 1
    sla_ms = float(cfg["sla_target_ms"])

    static_assert_guards = [
        f'static_assert(HW_DMA_ALIGN_BYTES == {align_b}, "SpecIR #E-401: Active DMA alignment must match {align_b}B");',
        f'static_assert((DMA_CR_BURST16 | DMA_CR_DIR_M2S | DMA_CR_EN) == 0x15, "SpecIR #E-401: Compliant DMA_CR burst mask must equal 0x15");',
        f'static_assert(((0x15u & (1u << {rsvd_bit})) == 0u), "SpecIR #E-401: DMA_CR must never assert Reserved Bit {rsvd_bit} ({rsvd_hex})");',
        f'static_assert(TILE_DIM == {num_banks}, "SpecIR #E-403: TILE_DIM must match {num_banks}-Bank interleaved crossbar width");',
        f'static_assert(((TILE_DIM + 1) % {num_banks}) == 1, "SpecIR #E-403: Coprime +1 padded row stride ({padded_stride}) required for 0% bank conflict");',
    ]

    return {
        "schema_version": "HardwareSpec-IR/v3.0",
        "ir_schema_version": "SpecMap-IR/v3.0",
        "document_id": "HX9000-RM-REV4.2-ERRATA-AWARE",
        "mmio_base_address": "0x40021000",
        "registers": {
            "DMA_CR": {
                "offset": "0x00",
                "address": "0x40021000",
                "access": "RW",
                "reset_value": "0x00000000",
                "rsvd_mbz_bit": rsvd_bit,
                "rsvd_mbz_mask_hex": rsvd_hex,
                "burst16_bit": 4,
                "burst16_mask_hex": "0x10",
                "compliant_single_beat_hex": "0x05",
                "compliant_burst16_hex": "0x15",
            },
            "DMA_SR": {
                "offset": "0x04",
                "address": "0x40021004",
                "access": "RO_VOLATILE",
                "done_bit": 0,
                "err_unaligned_bit": 2,
                "err_axi_deadlock_bit": 3,
                "mandatory_poll_api": "hw_poll_status_volatile()",
            },
            "DMA_SRC_ADDR": {
                "offset": "0x08",
                "address": "0x40021008",
                "alignment_bytes": align_b,
            },
            "DMA_DST_ADDR": {
                "offset": "0x0C",
                "address": "0x4002100C",
                "alignment_bytes": align_b,
            },
        },
        "dma_constraints": {
            "required_align_bytes": align_b,
            "address_mask_hex": f"0x{align_b - 1:02X}",
            "required_cpp_qualifier": f"alignas({align_b})",
            "required_base_offset": 0,
            "burst_beats": 16,
        },
        "timing_relations": {
            "core_clock_mhz": 400,
            "axi_clock_mhz": 200,
            "dmb_before_dma_trigger": True,
            "volatile_status_poll_required": True,
            "sla_frame_latency_ms": sla_ms,
            "wcet_ceiling_ms": sla_ms,
            "jitter_ceiling_us": 25.0,
            "thermal_ceiling_c": 75.0,
            "vdd_brownout_floor_mv": 820.0,
        },
        "sram_bank_constraints": {
            "num_banks": num_banks,
            "bank_width_bytes": 4,
            "unpadded_stride_floats": num_banks,
            "unpadded_stride_bytes": num_banks * 4,
            "mandatory_padded_stride_floats": padded_stride,
            "mandatory_padded_stride_bytes": padded_stride * 4,
            "coprime_gcd": math.gcd(padded_stride, num_banks),
        },
        "errata_overrides": [
            {
                "errata_id": "#E-401",
                "title": "AXI4 Arbiter Latch-Up & PDN di/dt Voltage Droop Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 610, 548, 685],
                "overrides_section": "Section 4.2.1 & 4.2.2",
                "override_target_page": 14,
                "defect_mechanism": (
                    f"Rev 4.2 silicon lacks unaligned DMA fixup microcode and latches DMA_CR[Bit {rsvd_bit}] ({rsvd_hex}) "
                    f"into an AXI4 arbiter retry oscillation (di/dt = 18.4 A/ns), drooping VDD_CORE to 682.4 mV."
                ),
                "mandatory_workaround": (
                    f"Enforce alignas({align_b}) at offset [0], clear Bit {rsvd_bit} ({rsvd_hex}), and set DMA_CR = 0x15 (BURST_16)."
                ),
                "static_assert_guard": static_assert_guards[0],
            },
            {
                "errata_id": "#E-402",
                "title": "L1 D-Cache Snoop Filter Bypass & Register Hoisting Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 688, 548, 752],
                "overrides_section": "Section 5.1.1 & 5.1.2",
                "override_target_page": 27,
                "defect_mechanism": (
                    "Hardware snoop filter on NET_DMB_SNOOP_INV fails to flush 56 dirty L1 cache lines during non-coherent DMA reads, "
                    "and -O2 optimization hoists non-volatile DMA_SR polling."
                ),
                "mandatory_workaround": (
                    "Call hw_dmb() immediately before hw_trigger_dma() and poll DMA_SR via hw_poll_status_volatile()."
                ),
                "static_assert_guard": static_assert_guards[1],
            },
            {
                "errata_id": "#E-403",
                "title": "Crossbar Wavefront Serialization & Bank 0 Thermal Runaway Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 755, 548, 830],
                "overrides_section": "Section 6.4.1 & 6.4.2",
                "override_target_page": 41,
                "defect_mechanism": (
                    f"Unpadded {num_banks}-float ({num_banks * 4}B) row stride serializes all {num_banks} vector lanes onto Bank 0, "
                    f"concentrating 45.9 W/mm² power density into Bank 0 sense amplifiers (97.7°C~99.2°C hotspot)."
                ),
                "mandatory_workaround": (
                    f"Override nominal stride with PADDED_STRIDE = TILE_DIM + 1 ({padded_stride} floats = {padded_stride * 4}B) so gcd({padded_stride}, {num_banks}) == 1."
                ),
                "static_assert_guard": static_assert_guards[4],
            },
        ],
        "symbol_traceability_map": [
            {
                "ir_symbol": "DMA_ALIGN_CONSTRAINT",
                "datasheet_ref": "Sec 4.2.1 (p.14) + Errata #E-401 (p.88)",
                "schematic_net": "NET_AXI_AWADDR[6:0]",
                "waveform_event_id": "EV-AXI-DEADLOCK-01",
                "cpp_ast_construct": f"alignas({align_b}) static float raw_backing_store[...]",
            },
            {
                "ir_symbol": "DMA_CR_RSVD_MBZ_GUARD",
                "datasheet_ref": f"Sec 4.2.2 (p.14) + Errata #E-401 (p.88) [Bit {rsvd_bit} {rsvd_hex}]",
                "schematic_net": "NET_DMA_CR_MBZ_TRAP",
                "waveform_event_id": "EV-AXI-DEADLOCK-01",
                "cpp_ast_construct": "DMA_CR_BURST16 | DMA_CR_DIR_M2S | DMA_CR_EN (0x15)",
            },
            {
                "ir_symbol": "L1_CACHE_COHERENCY_FENCE",
                "datasheet_ref": "Sec 5.1.1 & 5.1.2 (p.27) + Errata #E-402 (p.88)",
                "schematic_net": "NET_DMB_SNOOP_INV",
                "waveform_event_id": "EV-L1-SNOOP-HAZARD-02",
                "cpp_ast_construct": "hw_dmb(); + hw_poll_status_volatile();",
            },
            {
                "ir_symbol": "SRAM_COPRIME_STRIDE_PADDING",
                "datasheet_ref": f"Sec 6.4.3 (p.41) + Errata #E-403 (p.88) [{num_banks}-Bank]",
                "schematic_net": "NET_SRAM_XBAR_ARB & DIE_BANK0_SENSE",
                "waveform_event_id": "EV-SRAM-WAVEFRONT-HOTSPOT-03",
                "cpp_ast_construct": f"static constexpr int PADDED_STRIDE = TILE_DIM + 1; ({padded_stride})",
            },
        ],
        "static_assert_guards": static_assert_guards,
    }


def analyze_oscilloscope_csv(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parses the 2,500-row oscilloscope CSV log, extracts quantitative anomaly signatures,
    and attaches both the WaveformQA Event-Time JSON and the Typed HardwareSpec IR.
    """
    ensure_oscilloscope_csv(2500)
    total_samples = 0
    min_vdd_mv = 9999.0
    min_vdd_ts = "0.00"
    max_temp_c = 0.0
    max_temp_ts = "0.00"
    max_dirty_lines = 0
    stale_cache_ts = "0.00"
    critical_events: List[Dict[str, Any]] = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_samples += 1
            vdd = float(row["vdd_core_mv"])
            temp = float(row["die_hotspot_temp_c"])
            dirty = int(row["l1_dirty_lines"])
            tag = row["event_tag"]

            if vdd < min_vdd_mv:
                min_vdd_mv = vdd
                min_vdd_ts = row["timestamp_us"]
            if temp > max_temp_c:
                max_temp_c = temp
                max_temp_ts = row["timestamp_us"]
            if dirty > max_dirty_lines and "NO_DMB" in tag:
                max_dirty_lines = dirty
                stale_cache_ts = row["timestamp_us"]

            if tag in (
                "TRAP_AXI_ARB_LOCK_AND_UNALIGNED_ADDR",
                "WARN_STALE_L1_CACHE_LINE_READ_NO_DMB",
                "THERMAL_THROTTLE_BANK0_HOTSPOT_98C",
            ):
                critical_events.append(row)

    dirty_val = max_dirty_lines or 56
    stale_ts_val = stale_cache_ts if stale_cache_ts != "0.00" else "418.00"

    event_time_json = build_waveform_event_time_json(
        spec_config=spec_config,
        min_vdd_ts=min_vdd_ts,
        min_vdd_mv=min_vdd_mv,
        stale_cache_ts=stale_ts_val,
        dirty_lines=dirty_val,
        peak_temp_ts=max_temp_ts,
        peak_temp_c=max_temp_c,
    )
    hardware_spec_ir = build_hardware_spec_ir(spec_config=spec_config)

    return {
        "csv_file": "golden_dataset/oscilloscope_telemetry_dump.csv",
        "svg_schematic": "golden_dataset/AXI_Bus_Timing_Schematic.svg",
        "schematic_file": "golden_dataset/AXI_Bus_Timing_Schematic.svg",
        "schematic_exists": SVG_PATH.exists(),
        "schematic_bytes": SVG_PATH.stat().st_size if SVG_PATH.exists() else 0,
        "total_samples": total_samples,
        "csv_total_rows": total_samples,
        "min_vdd_core_mv": min_vdd_mv,
        "min_vdd_timestamp_us": min_vdd_ts,
        "max_dirty_cache_lines": dirty_val,
        "stale_cache_timestamp_us": stale_ts_val,
        "peak_die_temp_c": max_temp_c,
        "peak_temp_timestamp_us": max_temp_ts,
        "critical_events": critical_events,
        "event_time_telemetry": event_time_json,
        "waveform_event_time_json": event_time_json,
        "hardware_spec_ir": hardware_spec_ir,
    }


def get_multimodal_summary(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return analyze_oscilloscope_csv(spec_config=spec_config)


def render_dynamic_datasheet(spec_config: Dict[str, Any]) -> str:
    """
    Renders the HX-9000 Hardware Reference Manual dynamically reflecting active
    Live Spec Mutations AND Section 9: Silicon Errata Sheet (Cross-Page Overrides #E-401..#E-403).
    """
    align_b = int(spec_config.get("dma_align_bytes", 64))
    rsvd_bit = int(spec_config.get("rsvd_mbz_bit", 7))
    num_banks = int(spec_config.get("num_banks", 8))
    sla_ms = float(spec_config.get("sla_target_ms", 3.50))

    align_mask = align_b - 1
    rsvd_hex = 1 << rsvd_bit
    stride_bytes = num_banks * 4
    padded_stride = num_banks + 1
    padded_bytes = padded_stride * 4

    return f"""# HX-9000 Edge AI Tensor & DMA Coprocessor Reference Manual (Dynamic Spec)
**Document ID:** HX9000-RM-REV4.2-MUTABLE
**Active Hardware Configuration:** `DMA_ALIGN={align_b}B` | `RSVD_MBZ=Bit {rsvd_bit} (0x{rsvd_hex:02X})` | `SRAM_BANKS={num_banks}` | `SLA_TARGET={sla_ms:.2f}ms`

---

## Section 4.2 DMA Alignment & Control Register Constraints (Page 14)

### 4.2.1 {align_b}-Byte Cache-Line Address Alignment Rule [CRITICAL]
Both `DMA_SRC_ADDR` and `DMA_DST_ADDR` **MUST be aligned to a {align_b}-byte boundary** (`address % {align_b} == 0`, i.e., `address & 0x{align_mask:02X} == 0`).
* **Hardware Fault Behavior:** Initiating a DMA transfer with an unaligned pointer or buffer alignment `< {align_b}` raises `ERR_DMA_UNALIGNED_{align_b}B` (`DMA_SR[2] = 1`).
* **C/C++ Implementation Requirement:** Buffers passed to the DMA engine MUST be declared with `alignas({align_b})` and indexed at base offset `0` (`&raw_backing_store[0]`), followed by `HW_REGISTER_ALIGN(raw_backing_store);`.

### 4.2.2 `DMA_CR` (Offset `0x00`) Bitfield Specification
* **Bit `{rsvd_bit}` (`0x{rsvd_hex:02X} RSVD_MBZ`)**: **Reserved Must-Be-Zero (Factory Debug Bus Lock).**
  Writing `1` to Bit `{rsvd_bit}` (`0x{rsvd_hex:02X}`) locks the AXI4 arbiter (`AXI_WREADY=0` at CSV `t=142.50us`, `VDD_CORE` droop to `682.4mV`) and triggers `ERR_AXI_BUS_DEADLOCK_BIT{rsvd_bit}`.
* **Bit `4` (`0x10 BURST_16`)**: **16-Beat AXI4 Burst Mode Enable.**
  Setting `DMA_CR = 0x15` (`DMA_CR_BURST16 | DMA_CR_DIR_M2S | DMA_CR_EN`) enables 16-beat bursts and clears Bit `{rsvd_bit}`.

---

## Section 5.1 Cache Coherency & MMIO Polling Protocol (Page 27)
* **5.1.1 Mandatory `hw_dmb()` Barrier:** Software MUST invoke `hw_dmb()` immediately before `hw_trigger_dma()`. Omitting `hw_dmb()` causes 56 dirty L1 cache lines to be bypassed (CSV `t=418.00us`), corrupting output tensor CRC32.
* **5.1.2 Volatile Status Polling:** Software MUST poll `DMA_SR` via `hw_poll_status_volatile()` instead of `hw_poll_status_cached()`.

---

## Section 6.4 {num_banks}-Bank Interleaved Local SRAM & Bank Conflict Avoidance (Page 41)
* **6.4.1 {num_banks}-Bank Architecture:** Local SRAM has **{num_banks} parallel 32-bit banks** (`Bank = word_index mod {num_banks}`). `TILE_DIM` is `{num_banks}`.
* **6.4.2 The {stride_bytes}-Byte Stride Trap:** Unpadded `float local_tile[{num_banks}][{num_banks}]` has row stride `{num_banks}` floats (`{stride_bytes}B`). Column gather accesses `({num_banks}*r + c) mod {num_banks} = c`, causing a **{num_banks}-way Bank 0 collision (94.12% conflict)** and heating Bank 0 sense amplifiers to **98.4°C** (CSV `t=1120.50us`).
* **6.4.3 Mandatory Fix (+1 Element Row Padding):** Declare `static constexpr int PADDED_STRIDE = TILE_DIM + 1;` (`{padded_stride}` floats = `{padded_bytes}B`) and `float local_tile[TILE_DIM][PADDED_STRIDE];`. Since `gcd({padded_stride}, {num_banks}) = 1`, column accesses map to `({padded_stride}*r + c) mod {num_banks}`, achieving **0.00% Bank Conflict**, cooling the die to **62.2°C**, and meeting the `< {sla_ms:.2f} ms` SLA.

---

## Section 9. Silicon Errata Sheet — Rev 4.2 Mandatory Cross-Page Overrides (Page 88)
* **Errata `#E-401` (Overrides Sec 4.2, Page 14):** Rev 4.2 silicon lacks unaligned fixup microcode and latches `DMA_CR[Bit {rsvd_bit}]` (`0x{rsvd_hex:02X}`) into an AXI4 arbiter retry oscillation (`NET_DMA_CR_MBZ_TRAP`), inducing a `18.4 A/ns` PDN $di/dt$ transient spike that droops `VDD_CORE` to `682.4 mV` at `t=142.50us`. **Workaround:** Enforce `alignas({align_b})` + `static_assert` and clear Bit `{rsvd_bit}` (`DMA_CR = 0x15`).
* **Errata `#E-402` (Overrides Sec 5.1, Page 27):** Hardware snoop filter on `NET_DMB_SNOOP_INV` fails to snoop `56` dirty L1 cache lines (`t=418.00us`) when `DMA_CR.EN` is triggered without an explicit `hw_dmb()` barrier, and `-O2` loop hoisting caches `DMA_SR` unless `hw_poll_status_volatile()` is used.
* **Errata `#E-403` (Overrides Sec 6.4, Page 41):** Unpadded `{num_banks}`-float (`{stride_bytes}B`) stride causes `{num_banks}`-way wavefront serialization on `NET_SRAM_XBAR_ARB` (`Bank 0`), concentrating `45.9 W/mm²` switching power density into Bank 0 (`97.7°C~99.2°C` at `t=1120.50us`) and violating WCET (`<= {sla_ms:.2f}ms`) & Jitter (`<= 25.0us`). **Workaround:** Override row stride to `PADDED_STRIDE = TILE_DIM + 1` (`{padded_stride}` floats) and verify with `static_assert(((TILE_DIM + 1) % {num_banks}) == 1)`.
"""
