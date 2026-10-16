import difflib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
    """Locate g++ executable on Windows / MSYS2 / PATH."""
    candidates = [
        r"C:\msys64\mingw64\bin\g++.exe",
        r"C:\msys64\ucrt64\bin\g++.exe",
        shutil.which("g++"),
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


def compile_and_run_kernel(
    kernel_code: str,
    step_id: str = "iter",
    spec_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compiles the candidate C++ kernel with hw_emulator_harness.cpp using g++ -O2
    and dynamic hardware spec macros (-DHW_*), then executes the binary in the sandbox.
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

    import re
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

    display_cmd = f"g++ -O2 -std=c++17 {' '.join(macro_flags)} -Igolden_dataset kernel_{step_id}.cpp hw_emulator_harness.cpp"

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
        return {
            "compiled": False,
            "spec_config": cfg,
            "compiler_cmd": display_cmd,
            "compiler_stderr": f"Compiler invocation failed: {exc}",
            "telemetry": {
                "status": "COMPILE_ERROR",
                "fault_code": "ERR_COMPILER_FAILURE",
                "fault_detail": str(exc),
                "spec_config": cfg,
                "crc_passed": False,
                "accuracy_pct": 0.0,
                "latency_ms": 0.0,
                "throughput_gops": 0.0,
                "bank_conflict_rate_pct": 0.0,
                "peak_die_temp_c": 45.0,
                "avg_die_temp_c": 45.0,
                "bank_access_hist": [0] * num_banks,
                "bank_conflict_hist": [0] * num_banks,
                "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
            },
        }

    compile_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if comp_proc.returncode != 0:
        return {
            "compiled": False,
            "compile_time_ms": compile_ms,
            "spec_config": cfg,
            "compiler_cmd": display_cmd,
            "compiler_stderr": comp_proc.stderr.strip(),
            "telemetry": {
                "status": "COMPILE_ERROR",
                "fault_code": "ERR_SYNTAX_OR_ABI",
                "fault_detail": comp_proc.stderr.strip()[:280],
                "spec_config": cfg,
                "crc_passed": False,
                "accuracy_pct": 0.0,
                "latency_ms": 0.0,
                "throughput_gops": 0.0,
                "bank_conflict_rate_pct": 0.0,
                "peak_die_temp_c": 45.0,
                "avg_die_temp_c": 45.0,
                "bank_access_hist": [0] * num_banks,
                "bank_conflict_hist": [0] * num_banks,
                "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
            },
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
            "throughput_gops": 0.0,
            "bank_conflict_rate_pct": 0.0,
            "peak_die_temp_c": 45.0,
            "avg_die_temp_c": 45.0,
            "bank_access_hist": [0] * num_banks,
            "bank_conflict_hist": [0] * num_banks,
            "die_thermal_grid_c": [[45.0] * 8 for _ in range(8)],
        }

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
    }
