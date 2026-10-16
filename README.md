# SiliconHeal: Autonomous Hardware-Spec Verification & C++ Kernel Self-Healing Agent

> **Track 2 (Hard Tech) · Gemini 4 Argon (`thinking_level="high"`) + `antigravity-preview-09-2026` Native C++ Verification Sandbox**

**SiliconHeal** takes a hardware accelerator/MCU reference manual (`HX-9000 Neural DMA & Matrix Coprocessor Rev 4.2`) alongside a buggy, bottlenecked C++ tensor kernel (`kernel_buggy.cpp`). Using long-horizon sandbox verification loops, it cross-references datasheet register/memory constraints, compiles and runs a cycle-accurate C++ hardware test harness (`g++ -O2`), and autonomously self-heals the kernel across 3 iterations until all hardware faults, L1 cache coherency errors, and 8-Bank SRAM memory conflicts are resolved.

---

## Key Quantitative Results (Before vs. After)

| Metric | Iteration 0 (`kernel_buggy.cpp`) | Iteration 1 (DMA Fixed) | Iteration 2 (DMB + `volatile`) | Iteration 3 (Optimized) |
| :--- | :--- | :--- | :--- | :--- |
| **Sandbox Status** | `HARDWARE_FAULT` (Red) | `DATA_CORRUPTION` (Amber) | `SLA_LATENCY_VIOLATION` | **`ALL_SLA_PASSED` (Green)** |
| **DMA Control (`DMA_CR`)** | `0x85` (Bit 7 `RSVD_MBZ` Fault) | `0x05` (Single-Beat) | `0x05` (Single-Beat) | **`0x15` (`BURST_16` Enabled)** |
| **Tensor CRC32 Accuracy** | `0.0%` (Bus Deadlock) | `12.5%` (Stale L1 Cache) | `100.0%` (`0x7A9E4921`) | **`100.0%` (`0x7A9E4921`)** |
| **8-Bank SRAM Conflict Rate** | Halted | `94.12%` (8-Way Bank 0 Collision) | `94.12%` (8-Way Bank 0 Collision) | **`0.00%` (`+1` Row Padding)** |
| **Frame Latency (SLA < 3.5ms)** | Halted | `46.51 ms` | `46.51 ms` | **`2.46 ms` (18.9x Speedup)** |
| **Effective Throughput** | `0.00 GOPS` | `1.44 GOPS` | `1.44 GOPS` | **`27.28 GOPS`** |

---

## Project Structure

- [`golden_dataset/HX9000_Reference_Manual.md`](golden_dataset/HX9000_Reference_Manual.md): Hardware Reference Manual containing MMIO memory map (`0x4002_1000`), 64-byte DMA alignment rules (Sec 4.2), L1 cache coherency `hw_dmb()` & `volatile` polling rules (Sec 5.1), and 8-Bank Interleaved Local SRAM bank conflict formulas (Sec 6.4).
- [`golden_dataset/hw_emulator_harness.hpp`](golden_dataset/hw_emulator_harness.hpp) & [`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp): Cycle-accurate C++ hardware register & 8-bank SRAM verification harness.
- [`golden_dataset/kernel_buggy.cpp`](golden_dataset/kernel_buggy.cpp): Initial C++ tensor transpose & scale kernel containing 3 hardware specification violations.
- [`golden_dataset/patches/`](golden_dataset/patches/): Step-by-step verified C++ patches (`iter1`, `iter2`, `iter3`).
- [`backend/sandbox_runner.py`](backend/sandbox_runner.py): Invokes native `g++ -O2` inside the sandbox to compile and execute candidate kernels and capture JSON hardware telemetry.
- [`backend/agent_loop.py`](backend/agent_loop.py): Gemini 4 Argon (`thinking_level="high"`) system prompt, datasheet citation mapper, and verification loop controller.
- [`backend/server.py`](backend/server.py): Zero-dependency HTTP server serving the interactive dashboard and live `g++` compilation API.
- [`frontend/index.html`](frontend/index.html): 3-column stage presentation dashboard (Left: Datasheet Spec & MMIO Register Bitfield Inspector / Center: Interactive Visual Diff & Live C++ Editor / Right: 8-Bank SRAM Conflict Visualizer & Speedup Telemetry).

---

## Quick Start (10 Seconds)

```powershell
# 1. Start the live native g++ sandbox server
python backend/server.py

# 2. Open http://127.0.0.1:8765 in your browser (or open frontend/index.html directly)
```
