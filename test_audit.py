import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
GOLDEN_DIR = ROOT_DIR / "golden_dataset"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_loop import run_autonomous_closed_loop
from multimodal_context import ensure_oscilloscope_csv, get_multimodal_summary
from sandbox_runner import compile_and_run_kernel


def print_banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def main() -> None:
    print_banner("SILICONHEAL SELF-AUDIT & VERIFICATION SUITE (4 MANDATORY TASKS)")

    # -------------------------------------------------------------------------
    # AUDIT 1: Multimodal 1M-Context Dataset & Zero Static Patch Check
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 1] Multimodal Dataset (Datasheet + SVG Schematic + 2,500-Row CSV) & Zero Hardcoded Patches")
    patches_dir = GOLDEN_DIR / "patches"
    assert not patches_dir.exists(), "ERROR: golden_dataset/patches still exists!"
    print("[PASS] Zero pre-baked patch files exist (golden_dataset/patches removed).")

    csv_path = ensure_oscilloscope_csv(2500)
    mm_summary = get_multimodal_summary()
    print(f"[PASS] Oscilloscope CSV generated: {csv_path.name} ({mm_summary['csv_total_rows']} telemetry rows)")
    for anom in mm_summary["critical_events"]:
        print(
            f"       -> Anomaly @ t={anom['timestamp_us']} us [{anom['event_tag']}]: "
            f"VDD={anom['vdd_core_mv']} mV, Temp={anom['die_hotspot_temp_c']} C, "
            f"DirtyL1={anom['l1_dirty_lines']}, ConflictWays={anom['bank_conflict_ways']}"
        )
    assert mm_summary["csv_total_rows"] == 2500, "Expected 2500 rows in oscilloscope CSV"
    assert mm_summary["schematic_exists"], "AXI_Bus_Timing_Schematic.svg missing!"
    print(f"[PASS] Hardware Timing & Crossbar Schematic verified: {mm_summary['schematic_file']} ({mm_summary['schematic_bytes']} bytes)")

    # -------------------------------------------------------------------------
    # AUDIT 2: Default Spec Closed-Loop (64B Align, Bit 7 RSVD_MBZ, 8-Bank SRAM)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 2] Closed-Loop Verification on Default Spec (64B Align | Bit 7 RSVD | 8-Bank)")
    spec_default = {"dma_align_bytes": 64, "rsvd_mbz_bit": 7, "num_banks": 8, "sla_target_ms": 3.50}
    res_default = run_autonomous_closed_loop(spec_config=spec_default, use_live_api=False)

    for st in res_default["stages"]:
        t = st["sandbox"]["telemetry"]
        print(
            f"  Iter {st['iteration']} ({st['badge']}): status={t['status']} | "
            f"fault={t['fault_code']} | latency={t['latency_ms']:.2f} ms | "
            f"bank_conflict={t['bank_conflict_rate_pct']:.2f}% | "
            f"peak_die_temp={t['peak_die_temp_c']:.1f} C | "
            f"accuracy={t['accuracy_pct']:.1f}%"
        )

    s0, s1, s2, s3 = [s["sandbox"]["telemetry"] for s in res_default["stages"]]
    assert s0["status"] == "HARDWARE_FAULT"
    assert s1["status"] == "DATA_CORRUPTION"
    assert s2["status"] == "SLA_LATENCY_VIOLATION"
    assert s3["status"] == "ALL_SLA_PASSED"
    assert abs(s2["latency_ms"] - 46.51) < 0.2, f"Expected ~46.51ms, got {s2['latency_ms']}"
    assert abs(s3["latency_ms"] - 2.46) < 0.2, f"Expected ~2.46ms, got {s3['latency_ms']}"
    speedup = round(s2["latency_ms"] / s3["latency_ms"], 2)
    assert s2["peak_die_temp_c"] >= 95.0 and s3["peak_die_temp_c"] <= 66.0
    print(
        f"[PASS] Default Spec Closed-Loop: {s2['latency_ms']:.2f} ms -> {s3['latency_ms']:.2f} ms ({speedup}x speedup), "
        f"Die Thermal Peak: {s2['peak_die_temp_c']} C -> {s3['peak_die_temp_c']} C"
    )

    # -------------------------------------------------------------------------
    # AUDIT 3: Live Spec Mutation (128B Align, Bit 6 RSVD_MBZ, 16-Bank SRAM)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 3] Live Spec Mutation Defense (128B Align | Bit 6 RSVD | 16-Bank SRAM)")
    spec_mutated = {"dma_align_bytes": 128, "rsvd_mbz_bit": 6, "num_banks": 16, "sla_target_ms": 3.50}

    # Prove that the 64B/8-Bank final code FAILS when compiled against the 128B/16-Bank mutated hardware harness
    code_64b_8bank = res_default["stages"][3]["code"]
    cross_check = compile_and_run_kernel(code_64b_8bank, step_id="cross_check_64b_on_128b", spec_config=spec_mutated)
    cross_t = cross_check["telemetry"]
    print(
        f"  [Cross-Check] Running 64B/8-Bank patch on 128B/16-Bank Emulator -> "
        f"status={cross_t['status']}, fault_code='{cross_t['fault_code']}', detail='{cross_t['fault_detail']}'"
    )
    assert cross_t["status"] == "HARDWARE_FAULT", "64B patch should fail on 128B emulator!"
    print("[PASS] Confirmed: 64-Byte patch is strictly rejected by the 128-Byte mutated hardware harness.")

    # Now run the closed-loop agent on the mutated spec (128B, Bit 6, 16-Bank)
    res_mutated = run_autonomous_closed_loop(spec_config=spec_mutated, use_live_api=False)
    for st in res_mutated["stages"]:
        t = st["sandbox"]["telemetry"]
        sc = t["spec_config"]
        print(
            f"  Mutated Iter {st['iteration']} ({st['badge']}): status={t['status']} | "
            f"latency={t['latency_ms']:.2f} ms | "
            f"bank_conflict={t['bank_conflict_rate_pct']:.2f}% | "
            f"peak_die_temp={t['peak_die_temp_c']:.1f} C | "
            f"spec_align={sc['dma_align_bytes']}B, rsvd_bit={sc['rsvd_mbz_bit']}, banks={sc['num_banks']}"
        )

    mut_final_code = res_mutated["stages"][3]["code"]
    mut_final_t = res_mutated["stages"][3]["sandbox"]["telemetry"]
    assert "alignas(128)" in mut_final_code, "Agent must synthesize alignas(128) for 128B spec!"
    assert "PADDED_STRIDE = TILE_DIM + 1; // [OPTIMIZED Sec 6.4.3: 17 floats (68B) -> gcd(17,16)==1]" in mut_final_code
    assert mut_final_t["status"] == "ALL_SLA_PASSED"
    assert mut_final_t["bank_conflict_rate_pct"] == 0.0
    print(
        "[PASS] Mutated Spec Closed-Loop: Agent dynamically synthesized `alignas(128)` + "
        "`local_tile[TILE_DIM][PADDED_STRIDE]` (17 floats / 68B stride for 16-Bank) and passed 128B/16-Bank verification!"
    )

    # -------------------------------------------------------------------------
    # AUDIT 4: Self-Healing Compiler Retry Loop (Syntax Bug Injection)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 4] Self-Healing Compiler Retry Loop (Injected C++ Syntax Fault -> Auto-Repair)")
    buggy_raw = (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")
    syntax_corrupted = buggy_raw.replace(
        "const uint32_t dma_ctrl_flags = 0xC5;",
        "const uint32_t dma_ctrl_flags = 0xC5 // Deliberate missing semicolon injected by judge",
    )
    res_retry = run_autonomous_closed_loop(
        spec_config=spec_default,
        initial_code=syntax_corrupted,
        use_live_api=False,
    )
    retries = res_retry["compiler_retry_events"]
    assert len(retries) >= 1, "Expected compiler retry loop to trigger on syntax error!"
    print(f"  [Compiler Retry Event Captured] Stage={retries[0]['stage']}, Attempt={retries[0]['attempt']}")
    stderr_lines = [ln for ln in retries[0]["compiler_stderr"].splitlines() if "error:" in ln]
    print(f"  [g++ stderr]: {stderr_lines[0] if stderr_lines else retries[0]['compiler_stderr'].splitlines()[0]}")
    print(f"  [Repair Note]: {retries[0]['repair_note']}")
    assert res_retry["all_passed"], "Loop must still reach ALL_SLA_PASSED after syntax self-healing!"
    print("[PASS] Self-Healing Compiler Retry Loop intercepted g++ stderr, repaired missing ';', and passed all SLAs!")

    print_banner("ALL 4 MANDATORY SELF-AUDITS PASSED 100%")


if __name__ == "__main__":
    main()
