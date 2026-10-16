import difflib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT_DIR / "golden_dataset"
BUILD_DIR = ROOT_DIR / "backend" / ".sandbox_build"


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


def compute_unified_diff(before_code: str, after_code: str, from_label: str = "kernel_prev.cpp", to_label: str = "kernel_patched.cpp") -> str:
    before_lines = before_code.splitlines(keepends=True)
    after_lines = after_code.splitlines(keepends=True)
    diff = difflib.unified_diff(before_lines, after_lines, fromfile=from_label, tofile=to_label, n=3)
    return "".join(diff)


def compile_and_run_kernel(kernel_code: str, step_id: str = "iter") -> Dict[str, Any]:
    """
    Compiles the candidate C++ kernel with hw_emulator_harness.cpp using g++ -O2
    and executes the resulting binary inside the local sandbox.
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    kernel_src_path = BUILD_DIR / f"kernel_{step_id}.cpp"
    exe_path = BUILD_DIR / f"sandbox_{step_id}.exe"
    harness_cpp = GOLDEN_DIR / "hw_emulator_harness.cpp"

    kernel_src_path.write_text(kernel_code, encoding="utf-8")

    gpp = find_gpp_compiler()
    gpp_dir = str(Path(gpp).parent)
    env = os.environ.copy()
    if os.path.isdir(gpp_dir):
        env["PATH"] = gpp_dir + os.pathsep + env.get("PATH", "")

    compile_cmd = [
        gpp,
        "-O2",
        "-std=c++17",
        f"-I{GOLDEN_DIR}",
        str(kernel_src_path),
        str(harness_cpp),
        "-o",
        str(exe_path),
    ]

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
            "compiler_cmd": " ".join(compile_cmd),
            "compiler_stderr": f"Compiler invocation failed: {exc}",
            "telemetry": {
                "status": "COMPILE_ERROR",
                "fault_code": "ERR_COMPILER_FAILURE",
                "fault_detail": str(exc),
                "crc_passed": False,
                "accuracy_pct": 0.0,
                "latency_ms": 0.0,
                "throughput_gops": 0.0,
                "bank_conflict_rate_pct": 0.0,
                "bank_access_hist": [0] * 8,
                "bank_conflict_hist": [0] * 8,
            },
        }

    compile_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    if comp_proc.returncode != 0:
        return {
            "compiled": False,
            "compile_time_ms": compile_ms,
            "compiler_cmd": " ".join(compile_cmd),
            "compiler_stderr": comp_proc.stderr.strip(),
            "telemetry": {
                "status": "COMPILE_ERROR",
                "fault_code": "ERR_SYNTAX_OR_ABI",
                "fault_detail": comp_proc.stderr.strip()[:240],
                "crc_passed": False,
                "accuracy_pct": 0.0,
                "latency_ms": 0.0,
                "throughput_gops": 0.0,
                "bank_conflict_rate_pct": 0.0,
                "bank_access_hist": [0] * 8,
                "bank_conflict_hist": [0] * 8,
            },
        }

    # Execute compiled binary inside sandbox
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
            "crc_passed": False,
            "accuracy_pct": 0.0,
            "latency_ms": 0.0,
            "throughput_gops": 0.0,
            "bank_conflict_rate_pct": 0.0,
            "bank_access_hist": [0] * 8,
            "bank_conflict_hist": [0] * 8,
        }

    return {
        "compiled": True,
        "compile_time_ms": compile_ms,
        "exec_wall_time_ms": exec_ms,
        "exit_code": run_proc.returncode,
        "compiler_cmd": f"g++ -O2 -std=c++17 -Igolden_dataset kernel_{step_id}.cpp hw_emulator_harness.cpp",
        "compiler_stderr": comp_proc.stderr.strip(),
        "raw_stdout": raw_stdout,
        "telemetry": telemetry,
    }
