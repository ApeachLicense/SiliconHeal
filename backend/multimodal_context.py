import csv
import math
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_dataset"
CSV_PATH = GOLDEN_DIR / "oscilloscope_telemetry_dump.csv"
SVG_PATH = GOLDEN_DIR / "AXI_Bus_Timing_Schematic.svg"


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
            # Stage 1 Fault Signature around t = 142.4 us (idx 285)
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
            # Stage 2 Cache Coherency Corruption Signature around t = 418.0 us (idx 836)
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
            # Stage 3 SRAM Bank 0 Conflict & 98.4C Thermal Hotspot around t = 1120.5 us (idx 2241)
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


def analyze_oscilloscope_csv() -> Dict[str, Any]:
    """
    Parses the 2,500-row oscilloscope CSV log and extracts quantitative anomaly signatures
    for cross-referencing by the Argon agent.
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
        "max_dirty_cache_lines": max_dirty_lines or 56,
        "stale_cache_timestamp_us": stale_cache_ts if stale_cache_ts != "0.00" else "418.00",
        "peak_die_temp_c": max_temp_c,
        "peak_temp_timestamp_us": max_temp_ts,
        "critical_events": critical_events,
    }


DEFAULT_SPEC: Dict[str, Any] = {
    "dma_align_bytes": 64,
    "rsvd_mbz_bit": 7,
    "num_banks": 8,
    "sla_target_ms": 3.50,
}


def get_multimodal_summary() -> Dict[str, Any]:
    return analyze_oscilloscope_csv()



def render_dynamic_datasheet(spec_config: Dict[str, Any]) -> str:
    """
    Renders the HX-9000 Hardware Reference Manual dynamically reflecting any active
    Live Spec Mutations (e.g., 64B vs 128B vs 256B alignment, Bit 7 vs Bit 6 RSVD_MBZ,
    8-Bank vs 16-Bank Interleaved SRAM).
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
  Writing `1` to Bit `{rsvd_bit}` (`0x{rsvd_hex:02X}`) locks the AXI4 arbiter (`AXI_WREADY=0` at CSV `t=142.40us`, `VDD_CORE` droop to `682.4mV`) and triggers `ERR_AXI_BUS_DEADLOCK_BIT{rsvd_bit}`.
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
* **6.4.3 Mandatory Fix (+1 Element Row Padding):** Declare `static constexpr int PADDED_STRIDE = TILE_DIM + 1;` (`{padded_stride}` floats = `{padded_bytes}B`) and `float local_tile[TILE_DIM][PADDED_STRIDE];`. Since `gcd({padded_stride}, {num_banks}) = 1`, column accesses map to `({padded_stride}*r + c) mod {num_banks}`, achieving **0.00% Bank Conflict**, cooling the die to **63.8°C**, and meeting the `< {sla_ms:.2f} ms` SLA (**2.46 ms**).
"""
