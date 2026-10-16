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
    Ensures the baseline fault capture CSV (`oscilloscope_telemetry_dump.csv`) exists.
    Contains 2,500 mixed-signal samples (0.5 us/sample) captured during baseline buggy execution.
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
            pc = "0x08001208"
            dma_cr = "0xC5"
            wvalid = 1
            wready = 0  # AXI Arbiter Locked
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
            pc = "0x0800124C"
            dma_cr = "0x05"
            wvalid = 1
            wready = 1
            vdd_mv = 815.0
            dirty_lines = 56  # Unflushed dirty cache lines
            temp_c = 71.2
            event_tag = (
                "WARN_STALE_L1_CACHE_LINE_READ_NO_DMB"
                if idx == 836
                else "DMA_BYPASS_DIRTY_L1_CACHE"
            )
        elif idx >= 1800:
            pc = "0x080013A0"
            dma_cr = "0x05"
            wvalid = 1
            wready = 1
            sram_bank = 0
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


def scan_csv_anomaly_windows(csv_path: Path) -> Dict[str, Any]:
    """
    Scans the CSV file row-by-row using numeric signal thresholds (not hardcoded row indices)
    to detect anomaly windows, durations, and extrema.
    """
    axi_lock_rows: List[Dict[str, Any]] = []
    l1_hazard_rows: List[Dict[str, Any]] = []
    sram_stall_rows: List[Dict[str, Any]] = []
    critical_events: List[Dict[str, Any]] = []

    total_samples = 0
    min_vdd_row: Optional[Dict[str, Any]] = None
    max_dirty_row: Optional[Dict[str, Any]] = None
    max_temp_row: Optional[Dict[str, Any]] = None

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_samples += 1
            wvalid = int(row["axi_wvalid"])
            wready = int(row["axi_wready"])
            vdd = float(row["vdd_core_mv"])
            dirty = int(row["l1_dirty_lines"])
            conflict_ways = int(row["bank_conflict_ways"])
            temp = float(row["die_hotspot_temp_c"])

            # 1. Detect AXI4 bus lockup: WVALID=1 while WREADY=0
            if wvalid == 1 and wready == 0:
                axi_lock_rows.append(row)
                if min_vdd_row is None or vdd < float(min_vdd_row["vdd_core_mv"]):
                    min_vdd_row = row

            # 2. Detect Non-Coherent DMA read with dirty L1 cache lines: WVALID=1, WREADY=1, dirty_lines > 0
            if wvalid == 1 and wready == 1 and dirty > 0 and conflict_ways <= 1:
                l1_hazard_rows.append(row)
                if max_dirty_row is None or dirty > int(max_dirty_row["l1_dirty_lines"]):
                    max_dirty_row = row

            # 3. Detect SRAM multi-way bank conflict & thermal buildup: conflict_ways > 1
            if conflict_ways > 1:
                sram_stall_rows.append(row)
                if max_temp_row is None or temp > float(max_temp_row["die_hotspot_temp_c"]):
                    max_temp_row = row

    def _window_stats(window_rows: List[Dict[str, Any]], clock_mhz: int) -> Dict[str, Any]:
        if not window_rows:
            return {"start_us": "0.00", "end_us": "0.00", "duration_us": 0.0, "duration_cycles": 0, "sample_count": 0}
        t0 = float(window_rows[0]["timestamp_us"])
        t1 = float(window_rows[-1]["timestamp_us"])
        # Duration from first sample to last sample in window
        dur_us = max(0.5, round(t1 - t0, 2)) if len(window_rows) > 1 else 0.5
        return {
            "start_us": f"{t0:.2f}",
            "end_us": f"{t1:.2f}",
            "duration_us": dur_us,
            "duration_cycles": int(round(dur_us * clock_mhz)),
            "sample_count": len(window_rows),
        }

    w_axi = _window_stats(axi_lock_rows, clock_mhz=200)
    w_l1 = _window_stats(l1_hazard_rows, clock_mhz=200)
    w_sram = _window_stats(sram_stall_rows, clock_mhz=400)

    if min_vdd_row is not None:
        ev1 = dict(min_vdd_row)
        ev1["window_start_us"] = float(w_axi["start_us"])
        ev1["window_end_us"] = float(w_axi["end_us"])
        ev1["sample_count"] = int(w_axi["sample_count"])
        critical_events.append(ev1)
    if max_dirty_row is not None:
        ev2 = dict(max_dirty_row)
        ev2["window_start_us"] = float(w_l1["start_us"])
        ev2["window_end_us"] = float(w_l1["end_us"])
        ev2["sample_count"] = int(w_l1["sample_count"])
        critical_events.append(ev2)
    if max_temp_row is not None:
        ev3 = dict(max_temp_row)
        ev3["window_start_us"] = float(w_sram["start_us"])
        ev3["window_end_us"] = float(w_sram["end_us"])
        ev3["sample_count"] = int(w_sram["sample_count"])
        critical_events.append(ev3)

    return {
        "total_samples": total_samples,
        "axi_lock_window": w_axi,
        "axi_lock_peak_row": min_vdd_row or {},
        "l1_hazard_window": w_l1,
        "l1_hazard_peak_row": max_dirty_row or {},
        "sram_stall_window": w_sram,
        "sram_stall_peak_row": max_temp_row or {},
        "critical_events": critical_events,
    }


def build_waveform_event_time_json(
    scanned: Dict[str, Any],
    spec_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts scanned CSV anomaly windows into timestamp-aligned fault descriptors.
    All timestamps, windows, and cycle durations come directly from `scan_csv_anomaly_windows()`.
    """
    cfg = dict(DEFAULT_SPEC)
    if spec_config:
        cfg.update(spec_config)
    align_b = int(cfg["dma_align_bytes"])
    rsvd_bit = int(cfg["rsvd_mbz_bit"])
    rsvd_hex = f"0x{1 << rsvd_bit:02X}"
    num_banks = int(cfg["num_banks"])

    w1 = scanned["axi_lock_window"]
    r1 = scanned["axi_lock_peak_row"]
    t1_us = r1.get("timestamp_us", w1["start_us"])
    vdd1_mv = float(r1.get("vdd_core_mv", 0.0))
    pc1 = r1.get("pc_addr", "0x08001208")
    cyc1 = int(w1["duration_cycles"])

    w2 = scanned["l1_hazard_window"]
    r2 = scanned["l1_hazard_peak_row"]
    t2_us = r2.get("timestamp_us", w2["start_us"])
    dirty2 = int(r2.get("l1_dirty_lines", 0))
    pc2 = r2.get("pc_addr", "0x0800124C")
    cyc2 = int(w2["duration_cycles"])

    w3 = scanned["sram_stall_window"]
    r3 = scanned["sram_stall_peak_row"]
    t3_us = r3.get("timestamp_us", w3["start_us"])
    temp3_c = float(r3.get("die_hotspot_temp_c", 0.0))
    ways3 = int(r3.get("bank_conflict_ways", num_banks))
    pc3 = r3.get("pc_addr", "0x080013A0")
    cyc3 = int(w3["duration_cycles"])

    events = [
        {
            "event_id": "EV-AXI-DEADLOCK-01",
            "timestamp_us": t1_us,
            "window_us": f"{w1['start_us']}..{w1['end_us']}",
            "duration_cycles": cyc1,
            "pc_addr": pc1,
            "source_mapping": "kernel.cpp:line_19",
            "normalized_predicate": (
                f"t={t1_us}us ({w1['start_us']}..{w1['end_us']}us): AXI_WVALID=1 & WREADY=0 for {cyc1} AXI cycles | "
                f"PC=kernel.cpp:line_19 ({pc1}) | DMA_CR={r1.get('dma_cr_hex', '0xC5')} (Bit {rsvd_bit}={rsvd_hex} set) & "
                f"ADDR_ALIGN_MOD_{align_b}!=0"
            ),
            "schematic_net": "NET_DMA_CR_MBZ_TRAP & NET_AXI_AWADDR[6:0]",
            "errata_override_id": "#E-401",
            "physical_causal_mechanism": (
                f"Unaligned DMA address trap + Reserved Bit {rsvd_bit} lockup stalls AXI4 write-response "
                f"for {cyc1} cycles, dropping VDD_CORE to {vdd1_mv:.1f} mV."
            ),
            "observed_metrics": {
                "vdd_core_mv": vdd1_mv,
                "axi_wvalid": 1,
                "axi_wready": 0,
                "stall_cycles": cyc1,
            },
        },
        {
            "event_id": "EV-L1-SNOOP-HAZARD-02",
            "timestamp_us": t2_us,
            "window_us": f"{w2['start_us']}..{w2['end_us']}",
            "duration_cycles": cyc2,
            "pc_addr": pc2,
            "source_mapping": "kernel.cpp:line_24",
            "normalized_predicate": (
                f"t={t2_us}us ({w2['start_us']}..{w2['end_us']}us): L1_DIRTY_LINES={dirty2} for {cyc2} AXI cycles | "
                f"PC=kernel.cpp:line_24 ({pc2}) | NON_COHERENT_DMA_READ_ACTIVE"
            ),
            "schematic_net": "NET_DMB_SNOOP_INV",
            "errata_override_id": "#E-402",
            "physical_causal_mechanism": (
                f"Rev 4.2 hardware snoop filter disabled (#E-402): {dirty2} dirty L1 cache lines remain "
                f"unflushed during DMA transfer while non-volatile status polling reads cached state."
            ),
            "observed_metrics": {
                "l1_dirty_lines": dirty2,
                "dmb_fence_asserted": False,
                "corrupted_words_ratio": 0.875,
            },
        },
        {
            "event_id": "EV-SRAM-WAVEFRONT-HOTSPOT-03",
            "timestamp_us": t3_us,
            "window_us": f"{w3['start_us']}..{w3['end_us']}",
            "duration_cycles": cyc3,
            "pc_addr": pc3,
            "source_mapping": "kernel.cpp:line_30",
            "normalized_predicate": (
                f"t={t3_us}us ({w3['start_us']}..{w3['end_us']}us): SRAM_BANK_SEL=0 & CONFLICT_WAYS={ways3} for {cyc3} core cycles | "
                f"PC=kernel.cpp:line_30 ({pc3}) | STRIDE={num_banks * 4}B_WAVEFRONT_SERIALIZATION"
            ),
            "schematic_net": "NET_SRAM_XBAR_ARB & DIE_BANK0_SENSE",
            "errata_override_id": "#E-403",
            "physical_causal_mechanism": (
                f"Unpadded SramTile<float, {num_banks}, {num_banks}> stride ({num_banks * 4}B) routes column-gather "
                f"lanes to Bank 0 (gcd({num_banks}, {num_banks}) = {num_banks}), causing {ways3}-way serialization "
                f"and {temp3_c:.1f}°C Bank 0 sense-amp hotspot."
            ),
            "observed_metrics": {
                "active_sram_bank": 0,
                "bank_conflict_ways": ways3,
                "die_hotspot_temp_c": temp3_c,
                "duration_core_cycles": cyc3,
            },
        },
    ]

    for ev in events:
        ev["timestamp_window_us"] = ev["window_us"]
        ev["signal_predicate"] = ev["normalized_predicate"]
        ev["pc_symbol"] = f"{ev['source_mapping']} ({ev['pc_addr']})"

    return {
        "schema": "CSV-Anomaly-Scanner/v3.1",
        "clock_domain_core_mhz": 400,
        "clock_domain_axi_mhz": 200,
        "total_raw_csv_samples": scanned["total_samples"],
        "normalized_events": events,
        "events": events,
    }


def build_hardware_spec_ir(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Builds a structured Hardware Constraint & Cross-Page Errata dictionary from the active
    hardware parameters (`dma_align_bytes`, `rsvd_mbz_bit`, `num_banks`, `sla_target_ms`),
    deriving bitmasks from bit positions and solving `gcd(cols, num_banks) == 1` for coprime tile width.
    """
    cfg = dict(DEFAULT_SPEC)
    if spec_config:
        cfg.update(spec_config)

    align_b = int(cfg["dma_align_bytes"])
    rsvd_bit = int(cfg["rsvd_mbz_bit"])
    rsvd_hex = f"0x{1 << rsvd_bit:02X}"
    num_banks = int(cfg["num_banks"])
    sla_ms = float(cfg["sla_target_ms"])

    # Derive DMA_CR bitmasks from bit positions: Bit 0 (EN), Bit 2 (DIR_M2S), Bit 4 (BURST_16)
    en_bit = 0
    dir_m2s_bit = 2
    burst16_bit = 4
    single_beat_mask = (1 << en_bit) | (1 << dir_m2s_bit)
    burst16_mask = single_beat_mask | (1 << burst16_bit)
    assert (burst16_mask & (1 << rsvd_bit)) == 0, "Derived burst16_mask collides with RSVD_MBZ bit"

    # Solve for smallest coprime column stride > num_banks satisfying gcd(cols, num_banks) == 1
    coprime_stride = num_banks + 1
    while math.gcd(coprime_stride, num_banks) != 1:
        coprime_stride += 1

    static_assert_guards = [
        f'static_assert(HW_DMA_ALIGN_BYTES == {align_b}, "Spec #E-401: Active DMA alignment must match {align_b}B");',
        f'static_assert((DMA_CR_BURST16 | DMA_CR_DIR_M2S | DMA_CR_EN) == 0x{burst16_mask:02X}, "Spec #E-401: Derived DMA_CR burst mask");',
        f'static_assert(((0x{burst16_mask:02X}u & (1u << {rsvd_bit})) == 0u), "Spec #E-401: DMA_CR must clear Reserved Bit {rsvd_bit} ({rsvd_hex})");',
        f'static_assert(TILE_DIM == {num_banks}, "Spec #E-403: TILE_DIM must match {num_banks}-Bank crossbar width");',
        f'static_assert(({coprime_stride} % {num_banks}) == 1, "Spec #E-403: Coprime row stride ({coprime_stride}) required for 0% bank conflict");',
    ]

    return {
        "schema_version": "HardwareSpec-IR/v3.1",
        "ir_schema_version": "HardwareSpec-IR/v3.1",
        "document_id": "HX9000-RM-REV4.2-ERRATA-AWARE",
        "mmio_base_address": "0x40021000",
        "registers": {
            "DMA_CR": {
                "offset": "0x00",
                "address": "0x40021000",
                "access": "RW",
                "reset_value": "0x00000000",
                "en_bit": en_bit,
                "dir_m2s_bit": dir_m2s_bit,
                "burst16_bit": burst16_bit,
                "rsvd_mbz_bit": rsvd_bit,
                "rsvd_mbz_mask_hex": rsvd_hex,
                "burst16_mask_hex": f"0x{1 << burst16_bit:02X}",
                "compliant_single_beat_hex": f"0x{single_beat_mask:02X}",
                "compliant_burst16_hex": f"0x{burst16_mask:02X}",
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
            "mandatory_padded_stride_floats": coprime_stride,
            "mandatory_padded_stride_bytes": coprime_stride * 4,
            "coprime_gcd": math.gcd(coprime_stride, num_banks),
        },
        "errata_overrides": [
            {
                "errata_id": "#E-401",
                "title": "AXI4 Write-Response Lockup & Burst Alignment Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 610, 548, 685],
                "overrides_section": "Section 4.2.1 & 4.2.2",
                "override_target_page": 14,
                "defect_mechanism": (
                    f"Rev 4.2 silicon lacks unaligned DMA fixup microcode and locks the AXI4 arbiter when "
                    f"DMA_CR[Bit {rsvd_bit}] ({rsvd_hex}) is asserted."
                ),
                "mandatory_workaround": (
                    f"Align staging buffer to {align_b} bytes at base offset [0], clear Bit {rsvd_bit}, "
                    f"and assert EN (Bit 0), DIR_MEM2SRAM (Bit 2), and BURST_16 (Bit 4)."
                ),
                "static_assert_guard": static_assert_guards[0],
            },
            {
                "errata_id": "#E-402",
                "title": "Hardware L1 Snoop Filter Disabled in Rev 4.2 Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 688, 548, 752],
                "overrides_section": "Section 5.1.1 & 5.1.2",
                "override_target_page": 27,
                "defect_mechanism": (
                    "Contrary to Section 5.1, the automatic hardware snoop filter on NET_DMB_SNOOP_INV is disabled in Rev 4.2 silicon, "
                    "and -O2 optimization hoists non-volatile DMA_SR polling."
                ),
                "mandatory_workaround": (
                    "Call hw_dmb() immediately before hw_trigger_dma() and poll DMA_SR via hw_poll_status_volatile()."
                ),
                "static_assert_guard": static_assert_guards[1],
            },
            {
                "errata_id": "#E-403",
                "title": "Crossbar Bank-Conflict Serialization Override",
                "source_section": "Section 9 (Silicon Errata Sheet)",
                "source_page": 88,
                "source_bbox": [52, 755, 548, 830],
                "overrides_section": "Section 6.4.1 & 6.4.2",
                "override_target_page": 41,
                "defect_mechanism": (
                    f"Unpadded SramTile<float, {num_banks}, {num_banks}> shares factor {num_banks} with N_BANKS={num_banks}, "
                    f"routing all {num_banks} column-gather lanes to the same bank."
                ),
                "mandatory_workaround": (
                    f"Select SramTile column stride Cols >= TILE_DIM coprime to {num_banks} (gcd(Cols, {num_banks}) == 1)."
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
                "cpp_ast_construct": f"(1<<{burst16_bit}) | (1<<{dir_m2s_bit}) | (1<<{en_bit})",
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
                "datasheet_ref": f"Sec 6.4 (p.41) + Errata #E-403 (p.88) [{num_banks}-Bank]",
                "schematic_net": "NET_SRAM_XBAR_ARB & DIE_BANK0_SENSE",
                "waveform_event_id": "EV-SRAM-WAVEFRONT-HOTSPOT-03",
                "cpp_ast_construct": f"SramTile<float, TILE_DIM, {coprime_stride}> local_tile;",
            },
        ],
        "static_assert_guards": static_assert_guards,
    }


def analyze_oscilloscope_csv(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Scans `oscilloscope_telemetry_dump.csv` using numeric signal thresholds and builds
    the event-time anomaly summary and hardware constraint IR.
    """
    csv_path = ensure_oscilloscope_csv(2500)
    scanned = scan_csv_anomaly_windows(csv_path)

    r1 = scanned["axi_lock_peak_row"]
    r2 = scanned["l1_hazard_peak_row"]
    r3 = scanned["sram_stall_peak_row"]

    min_vdd_mv = float(r1.get("vdd_core_mv", 0.0))
    min_vdd_ts = r1.get("timestamp_us", "0.00")
    dirty_val = int(r2.get("l1_dirty_lines", 0))
    stale_ts_val = r2.get("timestamp_us", "0.00")
    max_temp_c = float(r3.get("die_hotspot_temp_c", 0.0))
    max_temp_ts = r3.get("timestamp_us", "0.00")

    resolved_cfg = dict(DEFAULT_SPEC)
    if spec_config:
        resolved_cfg.update(spec_config)

    event_time_json = build_waveform_event_time_json(
        scanned=scanned,
        spec_config=resolved_cfg,
    )
    hardware_spec_ir = build_hardware_spec_ir(spec_config=resolved_cfg)

    return {
        "csv_file": "golden_dataset/oscilloscope_telemetry_dump.csv",
        "svg_schematic": "golden_dataset/AXI_Bus_Timing_Schematic.svg",
        "schematic_file": "golden_dataset/AXI_Bus_Timing_Schematic.svg",
        "schematic_exists": SVG_PATH.exists(),
        "schematic_bytes": SVG_PATH.stat().st_size if SVG_PATH.exists() else 0,
        "total_samples": scanned["total_samples"],
        "csv_total_rows": scanned["total_samples"],
        "axi_wready_stall_samples": int(scanned["axi_lock_window"]["sample_count"]),
        "axi_wready_stall_cycles": int(scanned["axi_lock_window"]["duration_cycles"]),
        "min_vdd_core_mv": min_vdd_mv,
        "min_vdd_timestamp_us": min_vdd_ts,
        "max_dirty_cache_lines": dirty_val,
        "stale_cache_timestamp_us": stale_ts_val,
        "max_conflict_ways": int(r3.get("bank_conflict_ways", 0)),
        "peak_die_temp_c": max_temp_c,
        "peak_temp_timestamp_us": max_temp_ts,
        "critical_events": scanned["critical_events"],
        "event_time_telemetry": event_time_json,
        "waveform_event_time_json": event_time_json,
        "hardware_spec_ir": hardware_spec_ir,
    }


def get_multimodal_summary(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return analyze_oscilloscope_csv(spec_config=spec_config)


def render_dynamic_datasheet(spec_config: Dict[str, Any]) -> str:
    """
    Renders the HX-9000 Hardware Reference Manual dynamically reflecting active
    hardware parameters (`dma_align_bytes`, `rsvd_mbz_bit`, `num_banks`, `sla_target_ms`)
    and Section 9: Silicon Errata Sheet (Cross-Page Overrides #E-401..#E-403).
    Does NOT include pre-computed hex masks (`0x15`) or literal code snippets (`TILE_DIM + 1`),
    requiring the agent to derive them from the bitfield table and coprime math condition.
    """
    align_b = int(spec_config.get("dma_align_bytes", 64))
    rsvd_bit = int(spec_config.get("rsvd_mbz_bit", 7))
    num_banks = int(spec_config.get("num_banks", 8))
    sla_ms = float(spec_config.get("sla_target_ms", 3.50))

    align_mask = align_b - 1
    rsvd_hex = 1 << rsvd_bit
    stride_bytes = num_banks * 4

    return f"""# HX-9000 Edge AI Tensor & DMA Coprocessor Reference Manual (Dynamic Spec)
**Document ID:** HX9000-RM-REV4.2-MUTABLE
**Active Hardware Parameters:** `DMA_ALIGN={align_b}B` | `RSVD_MBZ=Bit {rsvd_bit} (0x{rsvd_hex:02X})` | `SRAM_BANKS={num_banks}` | `SLA_TARGET={sla_ms:.2f}ms`

---

## Section 4.2 DMA Alignment & Control Register Specification (Page 14)

### 4.2.1 {align_b}-Byte Cache-Line Address Alignment Rule
Both `DMA_SRC_ADDR` and `DMA_DST_ADDR` **must be aligned to a {align_b}-byte boundary** (`address % {align_b} == 0`, mask `0x{align_mask:02X}`).
* **Fault Behavior:** Initiating a DMA transfer with an unaligned pointer or buffer alignment `< {align_b}` raises `ERR_DMA_UNALIGNED_{align_b}B` (`DMA_SR[2] = 1`).
* **HAL Buffer Requirement:** Staging buffers must be aligned to `{align_b}` bytes and passed from base index `[0]` after `HW_REGISTER_ALIGN(...)`.

### 4.2.2 `DMA_CR` (Offset `0x00`) Bitfield Table
| Bit Position | Field Name | Hardware Function |
| :--- | :--- | :--- |
| `31:8` | `RESERVED` | Reserved; write as `0`. |
| **`{rsvd_bit}`** | **`RSVD_MBZ`** | **Reserved Must-Be-Zero (`0x{rsvd_hex:02X}`).** See Section 9 Errata `#E-401`. |
| **`4`** | **`BURST_16`** | `1` = 16-beat pipelined AXI4 burst mode; `0` = Single-beat mode (13-cycle bus turn-around penalty per beat). |
| **`2`** | **`DIR_MEM2SRAM`** | `1` = Transfer from System DRAM to Local Staging Buffer. |
| **`0`** | **`EN`** | `1` = Enable and start DMA channel. |

---

## Section 5.1 Nominal Cache Coherency & Status Polling (Page 27)
> **NOTE:** See **Section 9 (Silicon Errata `#E-402`, Page 88)** for critical Rev 4.2 silicon limitations that override this section.
* **5.1.1 Nominal Hardware Auto-Snoop:** The nominal specification includes a hardware L1 cache snoop port; an explicit barrier helper `hw_dmb()` is also provided in the HAL.
* **5.1.2 Status Register Polling:** The HAL provides `hw_poll_status_cached()` and `hw_poll_status_volatile()`.

---

## Section 6.4 {num_banks}-Bank Interleaved Local SRAM Architecture (Page 41)
* **6.4.1 {num_banks}-Bank Word Interleaving:** Local scratchpad tiles (`SramTile<float, Rows, Cols>`) map element `(r, c)` at word offset `w = r * Cols + c` to bank:
  `Bank_ID(r, c) = (r * Cols + c) mod {num_banks}`
* **6.4.2 Column-Gather Access Pattern:** During column-wise transpose reads (`r = 0..{num_banks - 1}` for fixed `c`), consecutive lanes step by `Cols` words (`{stride_bytes}B` when `Cols == {num_banks}`).

---

## Section 9. Silicon Errata Sheet — Rev 4.2 Mandatory Cross-Page Overrides (Page 88)
* **Errata `#E-401` (Overrides Sec 4.2, Page 14):** Rev 4.2 silicon lacks unaligned fixup microcode and locks the AXI4 write-response state machine (`AWVALID=1, AWREADY=0`) whenever `DMA_CR[Bit {rsvd_bit}]` (`RSVD_MBZ`) is set to `1`. Software must align the staging buffer to `{align_b}` bytes at offset `[0]`, clear Bit `{rsvd_bit}` (`0`), and assert `EN` (Bit 0), `DIR_MEM2SRAM` (Bit 2), and `BURST_16` (Bit 4).
* **Errata `#E-402` (Overrides Sec 5.1, Page 27):** Contrary to Section 5.1.1, the hardware L1 cache snoop filter on `NET_DMB_SNOOP_INV` is disabled in Rev 4.2 silicon due to a timing race. Software must call `hw_dmb()` immediately before `hw_trigger_dma()` and must poll `DMA_SR` via `hw_poll_status_volatile()`.
* **Errata `#E-403` (Overrides Sec 6.4, Page 41):** When `SramTile<float, Rows, Cols>` uses a column dimension `Cols` that shares a common factor with `{num_banks}` (`gcd(Cols, {num_banks}) > 1`), column-wise gather serializes across `gcd(Cols, {num_banks})` ways on `Bank 0`. Software must select a column width `Cols > {num_banks}` such that `gcd(Cols, {num_banks}) == 1` (coprime to `{num_banks}`).
"""
