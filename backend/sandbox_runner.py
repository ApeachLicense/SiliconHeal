import difflib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_dataset"
BUILD_DIR = ROOT_DIR / "backend" / ".sandbox_build"

DEFAULT_SPEC_CONFIG: Dict[str, Any] = {
    "dma_align_bytes": 64,
    "rsvd_mbz_bit": 7,
    "num_banks": 8,
    "sla_target_ms": 3.50,
}


def normalize_spec_config(spec_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_SPEC_CONFIG)
    if spec_config:
        if "dma_align_bytes" in spec_config:
            cfg["dma_align_bytes"] = int(spec_config["dma_align_bytes"])
        if "rsvd_mbz_bit" in spec_config:
            cfg["rsvd_mbz_bit"] = int(spec_config["rsvd_mbz_bit"])
        if "num_banks" in spec_config:
            cfg["num_banks"] = int(spec_config["num_banks"])
        if "sla_target_ms" in spec_config:
            cfg["sla_target_ms"] = float(spec_config["sla_target_ms"])
    return cfg


def find_gpp_compiler() -> str:
    """Locate g++ executable on Linux / macOS / Windows / MSYS2 / PATH."""
    candidates = [
        shutil.which("g++"),
        r"C:\msys64\mingw64\bin\g++.exe",
        r"C:\msys64\ucrt64\bin\g++.exe",
        shutil.which("clang++"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return str(c)
    return "g++"


def compute_unified_diff(
    before_code: str,
    after_code: str,
    from_label: str = "kernel_prev.cpp",
    to_label: str = "kernel_patched.cpp",
) -> str:
    before_lines = before_code.splitlines(keepends=True)
    after_lines = after_code.splitlines(keepends=True)
    diff = difflib.unified_diff(before_lines, after_lines, fromfile=from_label, tofile=to_label, n=3)
    return "".join(diff)


def _find_line_number(lines: List[str], pattern: str, default_line: int = 1) -> int:
    for idx, line in enumerate(lines, start=1):
        if pattern in line:
            return idx
    return default_line


def build_semantic_micro_profile(
    kernel_code: str,
    telemetry: Dict[str, Any],
    spec_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Instrumented Cycle & Bank-Conflict Profiler:
    Maps source C++ line numbers <-> observed SramTile<T, R, C> stride <-> wavefront/bank-conflict
    serialization ways <-> hardware stall causes.
    """
    lines = kernel_code.splitlines()
    num_banks = int(spec_config.get("num_banks", 8))
    align_b = int(spec_config.get("dma_align_bytes", 64))
    rsvd_bit = int(spec_config.get("rsvd_mbz_bit", 7))

    line_dma_buf = _find_line_number(lines, "raw_backing_store", 10)
    line_dma_ctrl = _find_line_number(lines, "dma_ctrl_flags", 16)
    line_dma_trig = _find_line_number(lines, "hw_trigger_dma", 19)
    line_dma_poll = _find_line_number(lines, "hw_poll_status", 24)
    line_sram_tile = _find_line_number(lines, "local_tile", 30)
    line_sram_gather = _find_line_number(lines, "dram_out[", 46)

    observed_cols = int(telemetry.get("observed_tile_cols", 0))
    sram_bypass = bool(telemetry.get("sram_bypass_detected", False))
    compiled_stride_floats = observed_cols if observed_cols > 0 else num_banks
    compiled_stride_bytes = compiled_stride_floats * 4
    stride_gcd = math.gcd(compiled_stride_floats, num_banks) if compiled_stride_floats > 0 else num_banks

    status = telemetry.get("status", "HARDWARE_FAULT")
    conflict_pct = float(telemetry.get("bank_conflict_rate_pct", 0.0))
    wavefront_ways = int(telemetry.get("wavefront_serialization_ways", num_banks if conflict_pct > 20 else 1))
    dst_aligned_ok = bool(telemetry.get("dst_aligned_ok", False))
    dmb_ok = bool(telemetry.get("dmb_barrier_called", False))
    vol_ok = bool(telemetry.get("volatile_polled", False))
    burst16_ok = bool(telemetry.get("burst16_enabled", False))

    if status == "COMPILE_ERROR":
        primary_stall = "COMPILER_SYNTAX_OR_STATIC_ASSERT_FAILURE"
    elif sram_bypass:
        primary_stall = "SRAM_TILE_INSTRUMENTATION_BYPASS_DETECTED"
    elif status == "HARDWARE_FAULT":
        primary_stall = "AXI4_ARBITER_DEADLOCK_AND_PDN_DI_DT_TRAP"
    elif status == "DATA_CORRUPTION":
        primary_stall = "L1_DCACHE_DIRTY_LINE_SNOOP_BYPASS_HAZARD"
    elif status == "SLA_LATENCY_VIOLATION":
        primary_stall = f"SRAM_CROSSBAR_{num_banks}WAY_WAVEFRONT_SERIALIZATION"
    else:
        primary_stall = "NONE_OPTIMAL_PIPELINED_EXECUTION"

    diagnostics = [
        {
            "hotspot_id": "MP-01-DMA-ALIGN-AND-MBZ",
            "source_location": f"kernel.cpp:{line_dma_buf},{line_dma_ctrl}",
            "hardware_unit": "AXI4_DMA_ENGINE & PDN_REGULATOR",
            "errata_ref": "#E-401 (Overrides Sec 4.2, p.88)",
            "compiled_property": f"alignas({telemetry.get('declared_dst_align', 16)}) | DMA_CR={telemetry.get('dma_cr_hex', '0x00')}",
            "wavefront_serialization_ratio": "DEADLOCK" if (status == "HARDWARE_FAULT" and not sram_bypass) else ("1.0x (16-Beat Burst)" if burst16_ok else "6.0x (Single-Beat Stall)"),
            "stall_contribution_pct": 100.0 if (status == "HARDWARE_FAULT" and not sram_bypass) else (26.4 if not burst16_ok else 3.8),
            "stall_reason": (
                f"Violates {align_b}B alignment or asserts DMA_CR Bit {rsvd_bit}"
                if (status == "HARDWARE_FAULT" and not sram_bypass)
                else ("Single-beat AXI transfers (BURST_16=0)" if not burst16_ok else f"Compliant {align_b}B aligned 16-beat burst")
            ),
            "passed": (status != "HARDWARE_FAULT") and dst_aligned_ok and burst16_ok,
        },
        {
            "hotspot_id": "MP-02-L1-SNOOP-COHERENCY",
            "source_location": f"kernel.cpp:{line_dma_trig},{line_dma_poll}",
            "hardware_unit": "L1_DCACHE_SNOOP_FILTER & MMIO_BUS",
            "errata_ref": "#E-402 (Overrides Sec 5.1, p.88)",
            "compiled_property": f"hw_dmb={dmb_ok} | volatile_poll={vol_ok} | dirty_lines={telemetry.get('dirty_l1_lines_bypassed', 56)}",
            "wavefront_serialization_ratio": "1.0x",
            "stall_contribution_pct": 0.0 if (dmb_ok and vol_ok) else 15.0,
            "stall_reason": (
                "56 dirty L1 cache lines bypassed on NET_DMB_SNOOP_INV & non-volatile -O2 poll"
                if not (dmb_ok and vol_ok)
                else "Explicit hw_dmb() clean-fence + hw_poll_status_volatile() active"
            ),
            "passed": dmb_ok and vol_ok and bool(telemetry.get("crc_passed", False)),
        },
        {
            "hotspot_id": "MP-03-SRAM-BANK-WAVEFRONT",
            "source_location": f"kernel.cpp:{line_sram_tile},{line_sram_gather}",
            "hardware_unit": f"{num_banks}_BANK_SRAM_CROSSBAR & BANK0_SENSE_AMP",
            "errata_ref": "#E-403 (Overrides Sec 6.4, p.88)",
            "compiled_property": (
                "BYPASSED_UNINSTRUMENTED_ARRAY"
                if sram_bypass
                else f"SramTile<float,{telemetry.get('observed_tile_rows', num_banks)},{compiled_stride_floats}> ({compiled_stride_bytes}B) | gcd({compiled_stride_floats},{num_banks})={stride_gcd}"
            ),
            "wavefront_serialization_ratio": (
                "BYPASS_FAULT"
                if sram_bypass
                else (f"{wavefront_ways}.0x ({wavefront_ways}-Way Bank Collision)" if conflict_pct > 5.0 else "1.0x (Conflict-Free Parallel)")
            ),
            "stall_contribution_pct": 100.0 if sram_bypass else (73.6 if conflict_pct > 5.0 else 0.0),
            "stall_reason": (
                "Kernel bypassed SramTile<T, R, C> instrumentation"
                if sram_bypass
                else (
                    f"Unpadded {compiled_stride_bytes}B stride routes {wavefront_ways} vector lanes to same bank ({conflict_pct:.2f}% conflict, {telemetry.get('peak_die_temp_c', 97.7):.1f}C hotspot)"
                    if conflict_pct > 5.0
                    else f"Coprime stride ({compiled_stride_floats} floats = {compiled_stride_bytes}B) achieves 0.00% conflict & {telemetry.get('peak_die_temp_c', 62.2):.1f}C uniform die temp"
                )
            ),
            "passed": (status == "ALL_SLA_PASSED") and (not sram_bypass) and (conflict_pct == 0.0),
        },
    ]

    for idx, d in enumerate(diagnostics):
        loc_parts = d["source_location"].split(":")[-1].split(",")
        d["line_number"] = int(loc_parts[0]) if loc_parts and loc_parts[0].isdigit() else (idx + 1)
        d["symbol_and_statement"] = f"{d['hotspot_id']} ({d['compiled_property']})"
        d["access_stride_bytes"] = compiled_stride_bytes if idx == 2 else (align_b if idx == 0 else 4)
        d["stall_root_cause"] = ("RESOLVED: " if d["passed"] else "STALL: ") + d["stall_reason"]

    return {
        "profiler_engine": "Instrumented-SramTile-Profiler/v3.1",
        "profiler_schema": "Instrumented-SramTile-Profiler/v3.1",
        "primary_stall_reason": primary_stall,
        "dominant_bottleneck": primary_stall,
        "compiled_stride_floats": compiled_stride_floats,
        "compiled_stride_bytes": compiled_stride_bytes,
        "wavefront_serialization_ways": wavefront_ways,
        "wavefront_serialization_ratio": f"{wavefront_ways}.0x",
        "static_assert_count": kernel_code.count("static_assert"),
        "line_level_diagnostics": diagnostics,
        "line_diagnostics": diagnostics,
    }


def compile_and_run_kernel(
    kernel_code: str,
    step_id: str = "iter",
    spec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compiles the candidate C++ kernel with hw_emulator_harness.cpp using g++ -O2
    and dynamic hardware spec macros (-DHW_*), forces POSIX C locale (`LC_ALL=C`)
    so compiler diagnostics are identical across Linux/macOS/Windows, and runs the binary.
    """
    cfg = normalize_spec_config(spec_config)
    align_b = cfg["dma_align_bytes"]
    rsvd_bit = cfg["rsvd_mbz_bit"]
    num_banks = cfg["num_banks"]
    sla_ms = cfg["sla_target_ms"]

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{step_id}_a{align_b}_b{rsvd_bit}_n{num_banks}"
    kernel_src_path = BUILD_DIR / f"kernel_{tag}.cpp"
    exe_path = BUILD_DIR / f"sandbox_{tag}.exe"
    harness_cpp = GOLDEN_DIR / "hw_emulator_harness.cpp"

    kernel_src_path.write_text(kernel_code, encoding="utf-8")

    gpp = find_gpp_compiler()
    gpp_dir = str(Path(gpp).parent)
    env = os.environ.copy()
    if os.path.isdir(gpp_dir):
        env["PATH"] = gpp_dir + os.pathsep + env.get("PATH", "")
    # Enforce POSIX C locale so Linux/macOS g++ emits standard ASCII quotes in stderr
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    m_align = re.search(r"alignas\s*\(\s*(\d+)\s*\)", kernel_code)
    kernel_declared_align = int(m_align.group(1)) if m_align else 16

    macro_flags = [
        f"-DHW_DMA_ALIGN_BYTES={align_b}",
        f"-DHW_RSVD_MBZ_BIT={rsvd_bit}",
        f"-DHW_NUM_BANKS={num_banks}",
        f"-DHW_SLA_TARGET_MS={sla_ms}",
        f"-DHW_KERNEL_DECLARED_ALIGN={kernel_declared_align}",
    ]

    compile_cmd = [
        gpp,
        "-O2",
        "-std=c++17",
        *macro_flags,
        f"-I{GOLDEN_DIR}",
        str(kernel_src_path),
        str(harness_cpp),
        "-o",
        str(exe_path),
    ]

    display_cmd = f"LC_ALL=C g++ -O2 -std=c++17 {' '.join(macro_flags)} -Igolden_dataset kernel_{step_id}.cpp hw_emulator_harness.cpp"

    t0 = time.perf_counter()
    try:
        comp_proc = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            env=env,
        )
    except Exception as exc:
        err_tel: Dict[str, Any] = {
            "status": "COMPILE_ERROR",
            "fault_code": "ERR_COMPILER_FAILURE",
            "fault_detail": str(exc),
            "spec_config": cfg,
            "crc_passed": False,
            "accuracy_pct": 0.0,
            "latency_ms": 0.0,
            "wcet_ms": 0.0,
            "jitter_us": 9999.0,
            "throughput_gops": 0.0,
            "bank_conflict_rate_pct": 0.0,
            "peak_die_temp_c": 45.0,
            "avg_die_temp_c": 45.0,
            "vdd_droop_mv": 850.0,
            "bank_access_hist": [0] * num_banks,
            "bank_conflict_hist": [0] * num_banks,
            "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
        }
        err_tel["semantic_micro_profile"] = build_semantic_micro_profile(kernel_code, err_tel, cfg)
        return {
            "compiled": False,
            "spec_config": cfg,
            "compiler_cmd": display_cmd,
            "compiler_stderr": f"Compiler invocation failed: {exc}",
            "telemetry": err_tel,
        }

    compile_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if comp_proc.returncode != 0:
        err_tel = {
            "status": "COMPILE_ERROR",
            "fault_code": "ERR_SYNTAX_OR_STATIC_ASSERT",
            "fault_detail": comp_proc.stderr.strip()[:280],
            "spec_config": cfg,
            "crc_passed": False,
            "accuracy_pct": 0.0,
            "latency_ms": 0.0,
            "wcet_ms": 0.0,
            "jitter_us": 9999.0,
            "throughput_gops": 0.0,
            "bank_conflict_rate_pct": 0.0,
            "peak_die_temp_c": 45.0,
            "avg_die_temp_c": 45.0,
            "vdd_droop_mv": 850.0,
            "bank_access_hist": [0] * num_banks,
            "bank_conflict_hist": [0] * num_banks,
            "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
        }
        err_tel["semantic_micro_profile"] = build_semantic_micro_profile(kernel_code, err_tel, cfg)
        return {
            "compiled": False,
            "compile_time_ms": compile_ms,
            "spec_config": cfg,
            "compiler_cmd": display_cmd,
            "compiler_stderr": comp_proc.stderr.strip(),
            "telemetry": err_tel,
        }

    t1 = time.perf_counter()
    run_proc = subprocess.run(
        [str(exe_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        env=env,
    )
    exec_ms = round((time.perf_counter() - t1) * 1000.0, 2)

    raw_stdout = run_proc.stdout.strip()
    try:
        telemetry = json.loads(raw_stdout)
    except Exception:
        telemetry = {
            "status": "RUNTIME_CRASH",
            "fault_code": f"EXIT_CODE_{run_proc.returncode}",
            "fault_detail": raw_stdout or run_proc.stderr.strip(),
            "spec_config": cfg,
            "crc_passed": False,
            "accuracy_pct": 0.0,
            "latency_ms": 0.0,
            "wcet_ms": 0.0,
            "jitter_us": 9999.0,
            "throughput_gops": 0.0,
            "bank_conflict_rate_pct": 0.0,
            "peak_die_temp_c": 45.0,
            "avg_die_temp_c": 45.0,
            "vdd_droop_mv": 850.0,
            "bank_access_hist": [0] * num_banks,
            "bank_conflict_hist": [0] * num_banks,
            "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
        }

    telemetry["semantic_micro_profile"] = build_semantic_micro_profile(kernel_code, telemetry, cfg)

    return {
        "compiled": True,
        "compile_time_ms": compile_ms,
        "exec_wall_time_ms": exec_ms,
        "exit_code": run_proc.returncode,
        "spec_config": cfg,
        "compiler_cmd": display_cmd,
        "compiler_stderr": comp_proc.stderr.strip(),
        "raw_stdout": raw_stdout,
        "telemetry": telemetry,
        "semantic_micro_profile": telemetry["semantic_micro_profile"],
    }
