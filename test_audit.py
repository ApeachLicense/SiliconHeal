import inspect
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
GOLDEN_DIR = ROOT_DIR / "golden_dataset"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_loop import (
    evaluate_six_stage_gates,
    repair_cpp_syntax_errors,
    run_autonomous_closed_loop,
)
from multimodal_context import (
    build_hardware_spec_ir,
    build_waveform_event_time_json,
    ensure_oscilloscope_csv,
    get_multimodal_summary,
)
from sandbox_runner import compile_and_run_kernel


def print_banner(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def main() -> None:
    print_banner("SILICONHEAL v3.1 SELF-AUDIT & ANTI-CHEAT VERIFICATION SUITE")

    # -------------------------------------------------------------------------
    # AUDIT 1: Multimodal Dataset, Zero Static Patches & Zero Answer Leaks in Datasheet
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 1] Multimodal Dataset, True CSV Signal Scanning & Zero Answer Leaks")
    patches_dir = GOLDEN_DIR / "patches"
    assert not patches_dir.exists(), "ERROR: golden_dataset/patches still exists!"
    print("[PASS] Zero pre-baked patch files exist (golden_dataset/patches removed).")

    manual_text = (GOLDEN_DIR / "HX9000_Reference_Manual.md").read_text(encoding="utf-8")
    for forbidden_leak in ("0x15", "TILE_DIM + 1", "PADDED_STRIDE", "46.51"):
        assert forbidden_leak not in manual_text, (
            f"Datasheet leak detected: '{forbidden_leak}' must not be hardcoded in HX9000_Reference_Manual.md!"
        )
    print("[PASS] Zero literal answer strings ('0x15', 'TILE_DIM + 1', 'PADDED_STRIDE', '46.51') in HX9000_Reference_Manual.md.")

    # Verify build_waveform_event_time_json has no default timestamp/cycle parameters
    sig = inspect.signature(build_waveform_event_time_json)
    for param_name, param in sig.parameters.items():
        assert param.default is inspect.Parameter.empty, (
            f"build_waveform_event_time_json parameter '{param_name}' must not have a hardcoded default!"
        )
    print("[PASS] build_waveform_event_time_json requires scanned CSV windows (zero default parameters).")

    csv_path = ensure_oscilloscope_csv(2500)
    mm_summary = get_multimodal_summary()
    print(f"[PASS] Oscilloscope CSV scanned: {csv_path.name} ({mm_summary['csv_total_rows']} telemetry rows)")
    assert mm_summary["axi_wready_stall_samples"] == 61 and mm_summary["axi_wready_stall_cycles"] > 0, (
        f"Expected 61 stall samples from CSV, got {mm_summary['axi_wready_stall_samples']}"
    )
    assert mm_summary["max_dirty_cache_lines"] == 56, f"Expected 56 dirty L1 lines from CSV, got {mm_summary['max_dirty_cache_lines']}"
    assert mm_summary["max_conflict_ways"] == 8, f"Expected 8 conflict ways from CSV, got {mm_summary['max_conflict_ways']}"
    for anom in mm_summary["critical_events"]:
        print(
            f"       -> Scanned Window @ t={anom['window_start_us']:.2f}..{anom['window_end_us']:.2f} us "
            f"(peak t={anom['timestamp_us']} us, {anom['sample_count']} rows): "
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
    assert res_default["mode"] == "offline_constraint_solver_fallback", (
        f"Expected honest offline mode label, got {res_default['mode']}"
    )

    for st in res_default["stages"]:
        t = st["sandbox"]["telemetry"]
        g = st["six_stage_gates"]
        print(
            f"  Iter {st['iteration']} ({st['badge']}): status={t['status']} | "
            f"gates={g['passed_count']}/{g['total_gates']} | "
            f"avg_lat={t['latency_ms']:.2f}ms (WCET={t['wcet_ms']:.2f}ms, Jitter=±{t['jitter_us']:.1f}us) | "
            f"conflict={t['bank_conflict_rate_pct']:.2f}% | "
            f"temp={t['peak_die_temp_c']:.1f}C | VDD={t['vdd_droop_mv']:.1f}mV | "
            f"SramTile_reads={t.get('sram_word_reads', 0)}"
        )

    s0, s1, s2, s3 = [s["sandbox"]["telemetry"] for s in res_default["stages"]]
    assert s0["status"] == "HARDWARE_FAULT"
    assert s1["status"] == "DATA_CORRUPTION"
    assert s2["status"] == "SLA_LATENCY_VIOLATION"
    assert s3["status"] == "ALL_SLA_PASSED"

    # Invariant-based assertions (driven by mechanistic cycle & bank-conflict counters)
    assert s2["sram_word_reads"] == 65536 and s3["sram_word_reads"] == 65536
    assert s2["bank_conflict_rate_pct"] > 80.0, f"Expected >80% conflict on unpadded 8-bank SramTile, got {s2['bank_conflict_rate_pct']}%"
    assert s3["bank_conflict_rate_pct"] == 0.0, f"Expected 0.0% conflict on coprime SramTile<float,8,9>, got {s3['bank_conflict_rate_pct']}%"
    assert s2["latency_ms"] > 25.0, f"Expected unoptimized latency > 25ms, got {s2['latency_ms']}"
    assert s3["latency_ms"] <= spec_default["sla_target_ms"], f"Expected optimized latency <= {spec_default['sla_target_ms']}ms, got {s3['latency_ms']}"
    speedup = round(s2["latency_ms"] / s3["latency_ms"], 2)
    assert speedup > 10.0, f"Expected >10x speedup from BURST_16 + coprime SramTile, got {speedup}x"
    assert s2["peak_die_temp_c"] >= 90.0 and s3["peak_die_temp_c"] <= 68.0
    print(
        f"[PASS] Default Spec Closed-Loop: {s2['latency_ms']:.2f} ms -> {s3['latency_ms']:.2f} ms ({speedup}x speedup), "
        f"Conflict: {s2['bank_conflict_rate_pct']:.2f}% -> {s3['bank_conflict_rate_pct']:.2f}%, "
        f"Die Thermal Peak: {s2['peak_die_temp_c']} C -> {s3['peak_die_temp_c']} C"
    )

    # -------------------------------------------------------------------------
    # AUDIT 3: Live Spec Mutation (128B Align, Bit 6 RSVD_MBZ, 16-Bank SRAM)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 3] Live Spec Mutation Defense (128B Align | Bit 6 RSVD | 16-Bank SRAM)")
    spec_mutated = {"dma_align_bytes": 128, "rsvd_mbz_bit": 6, "num_banks": 16, "sla_target_ms": 3.50}

    code_64b_8bank = res_default["stages"][3]["code"]
    cross_check = compile_and_run_kernel(code_64b_8bank, step_id="cross_check_64b_on_128b", spec_config=spec_mutated)
    if not cross_check["compiled"]:
        print(
            f"  [Cross-Check] Compile-Time SpecIR static_assert rejected 64B patch on 128B harness! "
            f"stderr='{cross_check['compiler_stderr'].splitlines()[0]}'"
        )
    else:
        cross_t = cross_check["telemetry"]
        assert cross_t["status"] == "HARDWARE_FAULT", "64B patch should fail on 128B emulator!"
    print("[PASS] Confirmed: 64-Byte patch is strictly rejected by the 128-Byte mutated SpecIR static_assert / harness.")

    res_mutated = run_autonomous_closed_loop(spec_config=spec_mutated, use_live_api=False)
    for st in res_mutated["stages"]:
        t = st["sandbox"]["telemetry"]
        sc = t["spec_config"]
        g = st["six_stage_gates"]
        print(
            f"  Mutated Iter {st['iteration']} ({st['badge']}): status={t['status']} | "
            f"gates={g['passed_count']}/{g['total_gates']} | "
            f"latency={t['latency_ms']:.2f}ms (WCET={t['wcet_ms']:.2f}ms) | "
            f"conflict={t['bank_conflict_rate_pct']:.2f}% | "
            f"spec={sc['dma_align_bytes']}B/Bit{sc['rsvd_mbz_bit']}/{sc['num_banks']}-Bank"
        )

    mut_final_code = res_mutated["stages"][3]["code"]
    mut_final_t = res_mutated["stages"][3]["sandbox"]["telemetry"]
    assert "alignas(128)" in mut_final_code, "Agent must synthesize alignas(128) for 128B spec!"
    assert "static_assert(DMA_BUF_ALIGN >= HW_DMA_ALIGN_BYTES" in mut_final_code
    assert "SramTile<float, TILE_DIM, COPRIME_COLS>" in mut_final_code
    assert mut_final_t["status"] == "ALL_SLA_PASSED"
    assert mut_final_t["bank_conflict_rate_pct"] == 0.0
    assert mut_final_t["observed_tile_cols"] == 17
    print(
        "[PASS] Mutated Spec Closed-Loop: Agent dynamically synthesized `alignas(128)` + compile-time `static_assert` "
        "+ `SramTile<float, 16, 17>` (coprime stride) and passed all 6 hard gates on 128B/16-Bank!"
    )

    # -------------------------------------------------------------------------
    # AUDIT 3B: Anti-Cheat Negative Verification (Raw-Array Bypass & Unpadded SramTile)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 3B] Anti-Cheat Negative Audit (Raw-Array Bypass & Unpadded SramTile Without Self-Report)")
    valid_opt_code = res_default["stages"][3]["code"]
    # CHEAT ATTEMPT 1: Replace instrumented SramTile with raw C array float local_tile[TILE_DIM][TILE_DIM]
    cheat_raw_array_kernel = (
        valid_opt_code.replace(
            "SramTile<float, TILE_DIM, COPRIME_COLS> local_tile;",
            "float local_tile[TILE_DIM][TILE_DIM]; // CHEAT: Raw C array bypassing SramTile instrumentation",
        )
        .replace("local_tile(r, c)", "local_tile[r][c]")
    )
    assert "float local_tile[TILE_DIM][TILE_DIM];" in cheat_raw_array_kernel

    cheat_res = compile_and_run_kernel(cheat_raw_array_kernel, step_id="audit_cheat_raw_array", spec_config=spec_default)
    assert cheat_res["compiled"], f"Cheat kernel should compile cleanly, stderr={cheat_res['compiler_stderr']}"
    cheat_tel = cheat_res["telemetry"]
    cheat_gates = evaluate_six_stage_gates(cheat_tel, spec_default, cheat_raw_array_kernel)
    print(
        f"  [Cheat Test 1 - Raw float local_tile[8][8]] status={cheat_tel['status']} | "
        f"fault_code={cheat_tel['fault_code']} | sram_reads={cheat_tel.get('sram_word_reads', 0)} | "
        f"gates={cheat_gates['passed_count']}/6"
    )
    assert cheat_tel["status"] == "HARDWARE_FAULT", f"Cheat kernel must fail with HARDWARE_FAULT, got {cheat_tel['status']}"
    assert cheat_tel["fault_code"] == "ERR_SRAM_TILE_BYPASS_DETECTED"
    assert cheat_tel["sram_bypass_detected"] is True
    assert cheat_gates["passed_count"] == 0 and not cheat_gates["all_gates_passed"]

    # Cheat Test 2: Unpadded SramTile<float, 8, 8> with BURST16 and NO self-reporting calls
    unpadded_sramtile_kernel = valid_opt_code.replace(
        "SramTile<float, TILE_DIM, COPRIME_COLS> local_tile;",
        "SramTile<float, TILE_DIM, TILE_DIM> local_tile; // Unpadded 8x8 SramTile with zero reporting calls",
    )
    unpadded_res = compile_and_run_kernel(unpadded_sramtile_kernel, step_id="audit_unpadded_sramtile", spec_config=spec_default)
    assert unpadded_res["compiled"], f"Unpadded SramTile kernel should compile cleanly, stderr={unpadded_res['compiler_stderr']}"
    unpadded_tel = unpadded_res["telemetry"]
    unpadded_gates = evaluate_six_stage_gates(unpadded_tel, spec_default, unpadded_sramtile_kernel)
    print(
        f"  [Cheat Test 2 - Unpadded SramTile<float,8,8> + BURST16] status={unpadded_tel['status']} | "
        f"conflict={unpadded_tel['bank_conflict_rate_pct']:.2f}% | latency={unpadded_tel['latency_ms']:.2f}ms | "
        f"gates={unpadded_gates['passed_count']}/6"
    )
    assert unpadded_tel["bank_conflict_rate_pct"] > 80.0, "Unpadded SramTile<float,8,8> must incur >80% bank conflict automatically!"
    assert unpadded_tel["status"] == "SLA_LATENCY_VIOLATION"
    assert not unpadded_gates["all_gates_passed"]
    print("[PASS] Anti-Cheat Verification Passed: Raw array bypass is blocked at Gate 1 (0/6), and unpadded SramTile<float,8,8> automatically incurs 87.50% bank conflict!")

    # -------------------------------------------------------------------------
    # AUDIT 4: Self-Healing Compiler Retry Loop (Linux Unicode Quotes + Live g++ Retry)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 4] Self-Healing Compiler Retry Loop (Linux Unicode Quotes & Live g++ Subprocess)")
    buggy_raw = (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")
    syntax_corrupted = buggy_raw.replace(
        "const uint32_t dma_ctrl_flags = 0xC5;",
        "const uint32_t dma_ctrl_flags = 0xC5 // Deliberate missing semicolon injected by judge",
    )

    # First explicitly test Linux UTF-8 locale curly quotes (‘,’ and ‘;’)
    linux_unicode_stderr = "kernel.cpp:31:5: error: expected \u2018,\u2019 or \u2018;\u2019 before \u2018if\u2019"
    repaired_from_unicode = repair_cpp_syntax_errors(syntax_corrupted, linux_unicode_stderr)
    assert "const uint32_t dma_ctrl_flags = 0xC5;" in repaired_from_unicode, (
        "repair_cpp_syntax_errors failed to handle Linux Unicode curly quotes (‘,’ or ‘;’)!"
    )
    print("[PASS] Linux UTF-8 curly-quote stderr (`expected ‘,’ or ‘;’ before ‘if’`) repaired accurately.")

    res_retry = run_autonomous_closed_loop(
        spec_config=spec_default,
        initial_code=syntax_corrupted,
        use_live_api=False,
    )
    retries = res_retry["compiler_retry_events"]
    assert len(retries) >= 1, "Expected compiler retry loop to trigger on syntax error!"
    print(f"  [Compiler Retry Event Captured] Stage={retries[0]['stage']}, Attempt={retries[0]['attempt']}")
    stderr_lines = [ln for ln in retries[0]["compiler_stderr"].splitlines() if "error:" in ln]
    print(f"  [g++ stderr (LC_ALL=C)]: {stderr_lines[0] if stderr_lines else retries[0]['compiler_stderr'].splitlines()[0]}")
    assert res_retry["all_passed"], "Loop must still reach ALL_SLA_PASSED after syntax self-healing!"
    print("[PASS] Self-Healing Compiler Retry Loop intercepted g++ stderr, repaired missing ';', and passed all 6 gates!")

    # -------------------------------------------------------------------------
    # AUDIT 5: HardwareSpec IR, Cross-Page Errata (#E-401..#E-403),
    #          CSV Event-Time JSON, Runtime Micro-Profiler & Fault Attribution
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 5] HardwareSpec IR + Cross-Page Errata + CSV Event-Time JSON + Micro-Profiler")
    spec_ir = res_default["hardware_spec_ir"]
    for req_key in ("registers", "dma_constraints", "timing_relations", "errata_overrides", "symbol_traceability_map", "static_assert_guards"):
        assert req_key in spec_ir, f"Missing key {req_key} in HardwareSpec IR!"
    errata_ids = [e["errata_id"] for e in spec_ir["errata_overrides"]]
    assert errata_ids == ["#E-401", "#E-402", "#E-403"], f"Unexpected errata IDs: {errata_ids}"
    print(f"[PASS] HardwareSpec IR Schema verified ({spec_ir['ir_schema_version']}) with Cross-Page Errata Overrides: {errata_ids}")

    wf_json = res_default["multimodal"]["waveform_event_time_json"]
    assert len(wf_json["events"]) == 3
    for ev in wf_json["events"]:
        print(f"       -> Scanned CSV Event [{ev['event_id']}] @ t={ev['timestamp_window_us']}: {ev['signal_predicate']} | Net={ev['schematic_net']}")

    mp_iter2 = res_default["stages"][2]["sandbox"]["semantic_micro_profile"]
    mp_iter3 = res_default["stages"][3]["sandbox"]["semantic_micro_profile"]
    print(f"[PASS] Runtime Counter Micro-Profiler verified: Iter 2 Bottleneck='{mp_iter2['dominant_bottleneck']}' -> Iter 3='{mp_iter3['dominant_bottleneck']}'")

    cg0 = s0["causal_graph"]
    cg2 = s2["causal_graph"]
    assert "voltage_droop_cause" in cg0 and "thermal_hotspot_cause" in cg2
    assert "telemetry_provenance" in s3, "Expected explicit telemetry_provenance metadata in sandbox output!"
    assert s0["pdn_di_dt_spike_a_per_ns"] > 15.0
    assert s2["bank0_power_density_w_mm2"] > 35.0
    assert s3["bank0_power_density_w_mm2"] < 10.0
    print(
        f"[PASS] Counter-Driven Physical Models & Provenance verified:\n"
        f"       1) Voltage Droop ({s0['vdd_droop_mv']} mV): PDN di/dt = {s0['pdn_di_dt_spike_a_per_ns']} A/ns\n"
        f"       2) Thermal Hotspot ({s2['peak_die_temp_c']} C -> {s3['peak_die_temp_c']} C): Bank 0 Power Density = {s2['bank0_power_density_w_mm2']} -> {s3['bank0_power_density_w_mm2']} W/mm^2"
    )

    # -------------------------------------------------------------------------
    # AUDIT 6: 6-Stage Hierarchical Hard Acceptance Gate & Verification Bundle
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 6] 6-Stage Hierarchical Hard Acceptance Gate Progression & Verification Bundle")
    gate_counts = [st["six_stage_gates"]["passed_count"] for st in res_default["stages"]]
    print(f"  Gate Progression across Iterations [0, 1, 2, 3]: {gate_counts} (out of 6)")
    assert gate_counts == [0, 1, 3, 6], f"Expected monotonic gate progression [0, 1, 3, 6], got {gate_counts}"

    for g in res_default["stages"][3]["six_stage_gates"]["gates"]:
        assert g["passed"], f"Final stage Gate {g['gate_id']} failed!"
        print(f"       -> [{g['gate_id']} PASS] {g['name']}: {g['measured']}")

    proof_bundle = res_default["proof_carrying_bundle"]
    assert proof_bundle["verification_status"] == "VERIFIED_6_OF_6_GATES"
    assert proof_bundle["execution_mode"] == "offline_constraint_solver_fallback"
    assert len(proof_bundle["datasheet_and_errata_citations"]) == 3
    assert len(proof_bundle["compile_time_static_asserts"]) == 3
    print(
        f"[PASS] Verification Evidence Bundle verified: schema={proof_bundle['bundle_schema']}, "
        f"mode={proof_bundle['execution_mode']}, status={proof_bundle['verification_status']}"
    )

    print_banner("ALL 7 MANDATORY SILICONHEAL v3.1 SELF-AUDITS (INCLUDING ANTI-CHEAT) PASSED 100%")


if __name__ == "__main__":
    main()
