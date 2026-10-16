import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Ensure backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
GOLDEN_DIR = ROOT_DIR / "golden_dataset"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agent_loop import (
    SILICON_HEAL_SYSTEM_PROMPT,
    build_stage_result,
    evaluate_six_stage_gates,
    run_autonomous_closed_loop,
)
from multimodal_context import (
    DEFAULT_SPEC,
    ensure_oscilloscope_csv,
    get_multimodal_summary,
)
from sandbox_runner import compile_and_run_kernel, compute_unified_diff

# Cache the initial default run for fast /api/init startup while computed from real g++ closed loop
_CACHED_DEFAULT_LOOP = None


def get_default_loop_result(force_refresh: bool = False) -> dict:
    global _CACHED_DEFAULT_LOOP
    if _CACHED_DEFAULT_LOOP is None or force_refresh:
        _CACHED_DEFAULT_LOOP = run_autonomous_closed_loop(
            spec_config=dict(DEFAULT_SPEC),
            use_live_api=None,
        )
    return _CACHED_DEFAULT_LOOP


class SiliconHealHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            index_file = FRONTEND_DIR / "index.html"
            if not index_file.exists():
                self.send_error(404, "frontend/index.html not found")
                return
            content = index_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/schematic.svg":
            svg_file = GOLDEN_DIR / "AXI_Bus_Timing_Schematic.svg"
            if not svg_file.exists():
                self.send_error(404, "AXI_Bus_Timing_Schematic.svg not found")
                return
            content = svg_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/telemetry.csv":
            csv_file = ensure_oscilloscope_csv(2500)
            content = csv_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/api/init":
            loop_res = get_default_loop_result()
            svg_text = (GOLDEN_DIR / "AXI_Bus_Timing_Schematic.svg").read_text(encoding="utf-8")
            csv_path = ensure_oscilloscope_csv(2500)
            csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
            preview_rows = [csv_lines[0]]
            for line in csv_lines[1:]:
                if any(
                    tag in line
                    for tag in (
                        "TRAP_AXI_ARB_LOCK_AND_UNALIGNED_ADDR",
                        "WARN_STALE_L1_CACHE_LINE_READ_NO_DMB",
                        "THERMAL_THROTTLE_BANK0_HOTSPOT_98C",
                    )
                ):
                    preview_rows.append(line)
            preview_rows.extend(csv_lines[280:300])
            self._send_json(
                {
                    "project": "SiliconHeal v3.1: Autonomous Hardware-Spec & Errata Verification Agent",
                    "engine": "Gemini API (gemini-2.5-pro) / Offline Constraint-Solver Fallback + g++ -O2 SramTile Sandbox",
                    "mode": loop_res["mode"],
                    "live_api_used": loop_res.get("live_api_used", False),
                    "spec_config": loop_res["spec_config"],
                    "manual_markdown": loop_res["manual_markdown"],
                    "multimodal": loop_res["multimodal"],
                    "hardware_spec_ir": loop_res["hardware_spec_ir"],
                    "proof_carrying_bundle": loop_res["proof_carrying_bundle"],
                    "schematic_svg": svg_text,
                    "csv_preview": "\n".join(preview_rows[:45]),
                    "csv_total_rows": len(csv_lines) - 1,
                    "system_prompt": SILICON_HEAL_SYSTEM_PROMPT,
                    "stages": loop_res["stages"],
                    "compiler_retry_events": loop_res["compiler_retry_events"],
                    "all_passed": loop_res["all_passed"],
                }
            )
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body_raw)
        except Exception:
            payload = {}

        if parsed.path == "/api/run-stage":
            iteration = int(payload.get("iteration", 0))
            spec_cfg = payload.get("spec_config") or dict(DEFAULT_SPEC)
            stage_data = build_stage_result(iteration, spec_config=spec_cfg)
            self._send_json({"ok": True, "stage": stage_data})
            return

        if parsed.path == "/api/mutate-and-heal":
            raw_spec = payload.get("spec_config") or {}
            spec_cfg = {
                "dma_align_bytes": int(raw_spec.get("dma_align_bytes", 64)),
                "rsvd_mbz_bit": int(raw_spec.get("rsvd_mbz_bit", 7)),
                "num_banks": int(raw_spec.get("num_banks", 8)),
                "sla_target_ms": float(raw_spec.get("sla_target_ms", 3.50)),
            }
            inject_syntax_bug = bool(payload.get("inject_syntax_bug", False))
            api_key = (payload.get("api_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
            use_live_api = payload.get("use_live_api")
            if use_live_api is None:
                use_live_api = bool(api_key)
            else:
                use_live_api = bool(use_live_api)
            model_name = payload.get("model_name") or "gemini-2.5-pro"

            initial_code = payload.get("initial_code")
            if not initial_code:
                initial_code = (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")

            if inject_syntax_bug:
                initial_code = initial_code.replace(
                    "const uint32_t dma_ctrl_flags = 0xC5;",
                    "const uint32_t dma_ctrl_flags = 0xC5 // [INJECTED SYNTAX BUG: missing semicolon]",
                )

            loop_res = run_autonomous_closed_loop(
                spec_config=spec_cfg,
                initial_code=initial_code,
                use_live_api=use_live_api,
                api_key=api_key,
                model_name=model_name,
            )
            self._send_json(
                {
                    "ok": True,
                    "mode": loop_res["mode"],
                    "live_api_used": loop_res.get("live_api_used", False),
                    "spec_config": loop_res["spec_config"],
                    "manual_markdown": loop_res["manual_markdown"],
                    "multimodal": loop_res["multimodal"],
                    "hardware_spec_ir": loop_res["hardware_spec_ir"],
                    "proof_carrying_bundle": loop_res["proof_carrying_bundle"],
                    "stages": loop_res["stages"],
                    "compiler_retry_events": loop_res["compiler_retry_events"],
                    "all_passed": loop_res["all_passed"],
                }
            )
            return

        if parsed.path == "/api/compile-custom":
            custom_code = payload.get("code", "")
            spec_cfg = payload.get("spec_config") or dict(DEFAULT_SPEC)
            baseline_code = (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")
            sandbox_res = compile_and_run_kernel(custom_code, step_id="custom", spec_config=spec_cfg)
            six_gates = evaluate_six_stage_gates(sandbox_res["telemetry"], spec_cfg, custom_code)
            diff_vs_baseline = compute_unified_diff(
                baseline_code,
                custom_code,
                from_label="kernel_buggy.cpp",
                to_label="kernel_custom.cpp",
            )
            self._send_json(
                {
                    "ok": True,
                    "spec_config": spec_cfg,
                    "sandbox": sandbox_res,
                    "six_stage_gates": six_gates,
                    "diff_vs_baseline": diff_vs_baseline,
                }
            )
            return

        self.send_error(404, "POST endpoint not found")


def run_server(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), SiliconHealHandler)
    print(f"[SiliconHeal] Server running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    run_server(port)
