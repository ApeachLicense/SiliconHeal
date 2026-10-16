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
    STAGE_METADATA,
    build_stage_result,
    call_gemini_live_patch,
)
from sandbox_runner import compile_and_run_kernel, compute_unified_diff


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

        if parsed.path == "/api/init":
            manual_md = (GOLDEN_DIR / "HX9000_Reference_Manual.md").read_text(encoding="utf-8")
            stages = [build_stage_result(i) for i in range(len(STAGE_METADATA))]
            self._send_json(
                {
                    "project": "SiliconHeal: Autonomous Hardware-Spec & Kernel Self-Healing Agent",
                    "engine": "Gemini 4 Argon (thinking_level=\"high\") + Antigravity Native g++ Sandbox",
                    "manual_markdown": manual_md,
                    "system_prompt": SILICON_HEAL_SYSTEM_PROMPT,
                    "stages": stages,
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
            stage_data = build_stage_result(iteration)
            self._send_json({"ok": True, "stage": stage_data})
            return

        if parsed.path == "/api/compile-custom":
            custom_code = payload.get("code", "")
            baseline_code = (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")
            sandbox_res = compile_and_run_kernel(custom_code, step_id="custom")
            diff_vs_baseline = compute_unified_diff(
                baseline_code,
                custom_code,
                from_label="kernel_buggy.cpp",
                to_label="kernel_custom.cpp",
            )
            self._send_json(
                {
                    "ok": True,
                    "sandbox": sandbox_res,
                    "diff_vs_baseline": diff_vs_baseline,
                }
            )
            return

        if parsed.path == "/api/live-gemini-heal":
            api_key = (payload.get("api_key") or os.environ.get("GEMINI_API_KEY") or "").strip()
            model_name = payload.get("model_name") or "gemini-2.5-pro"
            current_code = payload.get("code") or (GOLDEN_DIR / "kernel_buggy.cpp").read_text(encoding="utf-8")
            manual_md = (GOLDEN_DIR / "HX9000_Reference_Manual.md").read_text(encoding="utf-8")
            cur_sandbox = compile_and_run_kernel(current_code, step_id="live_in")

            if api_key:
                llm_res = call_gemini_live_patch(
                    api_key=api_key,
                    model_name=model_name,
                    manual_text=manual_md,
                    current_code=current_code,
                    telemetry=cur_sandbox["telemetry"],
                )
                if llm_res and "patched_code" in llm_res:
                    patched_code = llm_res["patched_code"]
                    new_sandbox = compile_and_run_kernel(patched_code, step_id="live_out")
                    diff_text = compute_unified_diff(current_code, patched_code)
                    self._send_json(
                        {
                            "ok": True,
                            "mode": "LIVE_GEMINI_API",
                            "llm_response": llm_res,
                            "sandbox": new_sandbox,
                            "diff": diff_text,
                        }
                    )
                    return

            # Fallback to deterministic stage 3 compiled live in g++ sandbox
            stage3 = build_stage_result(3)
            self._send_json({"ok": True, "mode": "SANDBOX_VERIFIED_FALLBACK", "stage": stage3})
            return

        self.send_error(404, "POST endpoint not found")


def run_server(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), SiliconHealHandler)
    print(f"[SiliconHeal] Server running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    run_server(port)
