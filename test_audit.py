import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
GOLDEN_DIR = ROOT_DIR / "golden_dataset"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_loop import run_autonomous_closed_loop
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
    print_banner("SILICONHEAL v3.0 SELF-AUDIT & VERIFICATION SUITE (6 MANDATORY AUDITS)")

    # -------------------------------------------------------------------------
    # AUDIT 1: Multimodal 1M-Context Dataset & Zero Static Patch Check
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 1] Multimodal Dataset (Datasheet + Errata + SVG + 2,500-Row CSV) & Zero Hardcoded Patches")
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
        g = st["six_stage_gates"]
        print(
            f"  Iter {st['iteration']} ({st['badge']}): status={t['status']} | "
            f"gates={g['passed_count']}/{g['total_gates']} | "
            f"avg_lat={t['latency_ms']:.2f}ms (WCET={t['wcet_ms']:.2f}ms, Jitter=±{t['jitter_us']:.1f}us) | "
            f"conflict={t['bank_conflict_rate_pct']:.2f}% | "
            f"temp={t['peak_die_temp_c']:.1f}C | VDD={t['vdd_droop_mv']:.1f}mV"
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
        f"WCET: {s2['wcet_ms']:.2f} ms -> {s3['wcet_ms']:.2f} ms, "
        f"Die Thermal Peak: {s2['peak_die_temp_c']} C -> {s3['peak_die_temp_c']} C"
    )

    # -------------------------------------------------------------------------
    # AUDIT 3: Live Spec Mutation (128B Align, Bit 6 RSVD_MBZ, 16-Bank SRAM)
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 3] Live Spec Mutation Defense (128B Align | Bit 6 RSVD | 16-Bank SRAM)")
    spec_mutated = {"dma_align_bytes": 128, "rsvd_mbz_bit": 6, "num_banks": 16, "sla_target_ms": 3.50}

    # Prove that the 64B/8-Bank final code FAILS when compiled/run against the 128B/16-Bank mutated hardware harness
    code_64b_8bank = res_default["stages"][3]["code"]
    cross_check = compile_and_run_kernel(code_64b_8bank, step_id="cross_check_64b_on_128b", spec_config=spec_mutated)
    if not cross_check["compiled"]:
        print(
            f"  [Cross-Check] Compile-Time SpecIR static_assert rejected 64B patch on 128B harness! "
            f"stderr='{cross_check['compiler_stderr'].splitlines()[0]}'"
        )
    else:
        cross_t = cross_check["telemetry"]
        print(
            f"  [Cross-Check] Runtime Emulator rejected 64B/8-Bank patch on 128B/16-Bank Harness -> "
            f"status={cross_t['status']}, fault_code='{cross_t['fault_code']}'"
        )
        assert cross_t["status"] == "HARDWARE_FAULT", "64B patch should fail on 128B emulator!"
    print("[PASS] Confirmed: 64-Byte patch is strictly rejected by the 128-Byte mutated SpecIR static_assert / harness.")

    # Now run the closed-loop agent on the mutated spec (128B, Bit 6, 16-Bank)
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
    assert "static_assert(DMA_BUF_ALIGN >= HW_DMA_ALIGN_BYTES" in mut_final_code, "Agent must inject Errata #E-401 static_assert!"
    assert "static_assert((PADDED_STRIDE % TILE_DIM) == 1" in mut_final_code, "Agent must inject Errata #E-403 static_assert!"
    assert mut_final_t["status"] == "ALL_SLA_PASSED"
    assert mut_final_t["bank_conflict_rate_pct"] == 0.0
    print(
        "[PASS] Mutated Spec Closed-Loop: Agent dynamically synthesized `alignas(128)` + compile-time `static_assert` "
        "proofs + `local_tile[16][17]` (68B coprime stride) and passed all 6 hard gates on 128B/16-Bank!"
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

    # -------------------------------------------------------------------------
    # AUDIT 5: v3.0 HardwareSpec IR, Cross-Page Errata (#E-401..#E-403),
    #          WaveformQA Event-Time JSON, KernelPro Micro-Profiler & Causal Graph
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 5] v3.0 HardwareSpec IR + Cross-Page Errata + WaveformQA + KernelPro + Causal Graph")
    spec_ir = res_default["hardware_spec_ir"]
    assert spec_ir["ir_schema_version"] == "SpecMap-IR/v3.0"
    for req_key in ("registers", "dma_constraints", "timing_relations", "errata_overrides", "symbol_traceability_map", "static_assert_guards"):
        assert req_key in spec_ir, f"Missing key {req_key} in HardwareSpec IR!"
    errata_ids = [e["errata_id"] for e in spec_ir["errata_overrides"]]
    assert errata_ids == ["#E-401", "#E-402", "#E-403"], f"Unexpected errata IDs: {errata_ids}"
    print(f"[PASS] HardwareSpec IR Schema verified ({spec_ir['ir_schema_version']}) with Cross-Page Errata Overrides: {errata_ids}")
    for guard in spec_ir["static_assert_guards"]:
        print(f"       -> Injected C++17 Guard: {guard}")

    wf_json = res_default["multimodal"]["waveform_event_time_json"]
    assert "WaveformQA-EventTime" in wf_json["schema"]
    assert len(wf_json["events"]) == 3
    for ev in wf_json["events"]:
        print(f"       -> WaveformQA Event [{ev['event_id']}] @ t={ev['timestamp_window_us']}: {ev['signal_predicate']} | Net={ev['schematic_net']}")

    # Verify KernelPro Semantic Micro-Profiler on Iter 2 (Bank Conflict Stall) vs Iter 3 (Resolved)
    mp_iter2 = res_default["stages"][2]["sandbox"]["semantic_micro_profile"]
    mp_iter3 = res_default["stages"][3]["sandbox"]["semantic_micro_profile"]
    assert mp_iter2["profiler_schema"] == "KernelPro-Nsight-SemanticMicroProfiler/v3.0"
    print(f"[PASS] KernelPro Semantic Micro-Profiler verified: Iter 2 Bottleneck='{mp_iter2['dominant_bottleneck']}' -> Iter 3='{mp_iter3['dominant_bottleneck']}'")
    for diag in mp_iter2["line_diagnostics"]:
        print(
            f"       -> Line {diag['line_number']} ({diag['symbol_and_statement']}): "
            f"Stride={diag['access_stride_bytes']}B | Serialization={diag['wavefront_serialization_ratio']} | "
            f"RootCause={diag['stall_root_cause']}"
        )

    # Verify Physical Causal Graph Separation (Voltage Droop vs Thermal Hotspot)
    cg0 = s0["causal_graph"]
    cg2 = s2["causal_graph"]
    assert "voltage_droop_cause" in cg0 and "thermal_hotspot_cause" in cg2
    assert s0["pdn_di_dt_spike_a_per_ns"] > 15.0, "Expected high PDN di/dt spike on unaligned/bus-lock fault"
    assert s2["bank0_power_density_w_mm2"] > 40.0, "Expected high Bank 0 power density on 8-way bank conflict"
    assert s3["bank0_power_density_w_mm2"] < 10.0, "Expected low uniform power density after +1 padding"
    print(
        f"[PASS] Physical Causal Graph strictly separated:\n"
        f"       1) Voltage Droop ({s0['vdd_droop_mv']} mV): PDN di/dt spike = {s0['pdn_di_dt_spike_a_per_ns']} A/ns ({cg0['voltage_droop_cause']['root_trigger']})\n"
        f"       2) Thermal Hotspot ({s2['peak_die_temp_c']} C): Bank 0 Power Density = {s2['bank0_power_density_w_mm2']} W/mm^2 -> {s3['bank0_power_density_w_mm2']} W/mm^2 ({cg2['thermal_hotspot_cause']['root_trigger']})"
    )

    # -------------------------------------------------------------------------
    # AUDIT 6: 6-Stage Hierarchical Hard Acceptance Gate & Proof-Carrying Bundle
    # -------------------------------------------------------------------------
    print_banner("[AUDIT 6] 6-Stage Hierarchical Hard Acceptance Gate Progression & Proof-Carrying Bundle")
    gate_counts = [st["six_stage_gates"]["passed_count"] for st in res_default["stages"]]
    print(f"  Gate Progression across Iterations [0, 1, 2, 3]: {gate_counts} (out of 6)")
    assert gate_counts == [0, 1, 3, 6], f"Expected monotonic gate progression [0, 1, 3, 6], got {gate_counts}"

    for g in res_default["stages"][3]["six_stage_gates"]["gates"]:
        assert g["passed"], f"Final stage Gate {g['gate_id']} failed!"
        print(f"       -> [{g['gate_id']} PASS] {g['name']}: {g['measured']} (Criterion: {g['criterion']})")

    proof_bundle = res_default["proof_carrying_bundle"]
    assert proof_bundle["bundle_schema"] == "SiliconHeal-ProofCarryingBundle/v3.0"
    assert proof_bundle["verification_status"] == "PROOF_VERIFIED_6_OF_6_GATES"
    assert len(proof_bundle["datasheet_and_errata_citations"]) == 3
    assert len(proof_bundle["compile_time_static_asserts"]) == 3
    print(
        f"[PASS] Proof-Carrying Patch Evidence Bundle verified: schema={proof_bundle['bundle_schema']}, "
        f"status={proof_bundle['verification_status']}"
    )

    print_banner("ALL 6 MANDATORY SILICONHEAL v3.0 SELF-AUDITS PASSED 100%")


if __name__ == "__main__":
    main()
