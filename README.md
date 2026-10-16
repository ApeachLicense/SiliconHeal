# SiliconHeal v2.0: Autonomous Hardware-Spec Verification & C++ Kernel Self-Healing Agent

> **Track 2 (Hard Tech) · Gemini 4 Argon (`thinking_level="high"`, 1M Multimodal Context) + `antigravity-preview-09-2026` Native C++ Verification Sandbox**

**SiliconHeal** ingests a multimodal hardware bundle—the **HX-9000 Reference Manual**, **AXI4 Bus Timing & PCB Schematic (`AXI_Bus_Timing_Schematic.svg`)**, and a **2,500-row Mixed-Signal Oscilloscope Telemetry Dump (`oscilloscope_telemetry_dump.csv`)**—alongside a buggy, bottlenecked C++ tensor kernel (`kernel_buggy.cpp`). Using closed-loop sandbox verification (`g++ -O2`), it cross-references datasheet constraints, schematic nets, and oscilloscope anomalies, dynamically synthesizes C++ repairs (with an automatic **`g++ stderr` compiler self-healing retry loop**), and resolves all hardware faults, L1 cache coherency hazards, and N-Bank SRAM memory conflicts.

---

## Quantitative Verification Results (Default Spec vs. Mutated Spec)

### 1. Default Spec (`64-Byte` DMA Align · `Bit 7` Reserved · `8-Bank` SRAM)

| Metric | Iter 0 (`kernel_buggy.cpp`) | Iter 1 (DMA Fixed) | Iter 2 (DMB + `volatile`) | Iter 3 (Optimized) |
| :--- | :--- | :--- | :--- | :--- |
| **Sandbox Status** | `HARDWARE_FAULT` (Red) | `DATA_CORRUPTION` (Amber) | `SLA_LATENCY_VIOLATION` | **`ALL_SLA_PASSED` (Green)** |
| **DMA Control (`DMA_CR`)** | `0x85` (Bit 7 `RSVD_MBZ` Fault) | `0x05` (Single-Beat) | `0x05` (Single-Beat) | **`0x15` (`BURST_16` Enabled)** |
| **Tensor CRC32 Accuracy** | `0.0%` (Bus Deadlock) | `12.5%` (Stale L1 Cache) | `100.0%` (`0x7A9E4921`) | **`100.0%` (`0x7A9E4921`)** |
| **8-Bank SRAM Conflict Rate** | Halted | `94.12%` (8-Way Bank 0 Collision) | `94.12%` (8-Way Bank 0 Collision) | **`0.00%` (`local_tile[8][9]`)** |
| **Peak Silicon Die Temp** | `94.2 °C` (Bus Contention) | `97.7 °C` (Bank 0 Hotspot) | `97.7 °C` (Bank 0 Hotspot) | **`62.2 °C` (Uniform Thermal)** |
| **Frame Latency (SLA ≤ 3.5ms)** | Halted | `46.51 ms` | `46.51 ms` | **`2.46 ms` (18.91x Speedup)** |
| **Effective Throughput** | `0.00 GOPS` | `1.44 GOPS` | `1.44 GOPS` | **`27.28 GOPS`** |

### 2. Live Mutated Spec (`128-Byte` DMA Align · `Bit 6` Reserved · `16-Bank` SRAM)

| Verification Scenario | Declared Align | Tile Stride | Bank Conflict Rate | Peak Die Temp | Frame Latency | Sandbox Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old `64B / 8-Bank` Patch on `128B / 16-Bank` HW** | `alignas(64)` | `local_tile[8][9]` | Halted (`DMA_ERR_BIT` + `RSVD_MBZ`) | `94.2 °C` | Halted | ❌ `HARDWARE_FAULT` |
| **Live Autonomous Synthesis on `128B / 16-Bank` HW** | **`alignas(128)`** | **`local_tile[16][17]`** | **`0.00%`** | **`62.2 °C`** | **`1.65 ms` (28.15x)** | ✅ **`ALL_SLA_PASSED`** |

---

## Core Architecture & 4 Key Capabilities

1. **True Closed-Loop Verification & Self-Healing Compiler Retry Loop ([`backend/agent_loop.py`](backend/agent_loop.py))**
   - Zero pre-baked static patch files (`golden_dataset/patches/` has been completely removed).
   - Every iteration parses active hardware constraints from the mutated datasheet, multimodal logs, and live `g++` sandbox JSON output to synthesize C++ code from scratch.
   - If custom/corrupted code or an LLM response triggers a `g++` syntax error, the **Self-Healing Compiler Retry Loop** captures `g++ stderr` line numbers and error tokens and automatically repairs the C++ AST before re-compiling.

2. **Live Chaos / Spec Mutator ([`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp) & [`frontend/index.html`](frontend/index.html))**
   - Interactive judge defense bar allows real-time mutation of:
     - **DMA Alignment (`-DHW_DMA_ALIGN_BYTES`)**: `64-Byte` / `128-Byte` / `256-Byte`
     - **Forbidden `DMA_CR` Bit (`-DHW_RSVD_MBZ_BIT`)**: `Bit 7 (0x80)` / `Bit 6 (0x40)`
     - **Local SRAM Banks (`-DHW_NUM_BANKS`)**: `8-Bank (16x16)` / `16-Bank (16x16)`
     - **Live Syntax Bug Injection**: Injects C++ syntax errors into the editor and demonstrates automatic recovery via `g++ stderr` feedback.

3. **1M-Token Multimodal Hardware Context ([`backend/multimodal_context.py`](backend/multimodal_context.py))**
   - [`golden_dataset/HX9000_Reference_Manual.md`](golden_dataset/HX9000_Reference_Manual.md): Full MMIO register map (`0x4002_1000`), DMA alignment rules, L1 cache coherency (`hw_dmb()`), and N-Bank SRAM conflict math.
   - [`golden_dataset/AXI_Bus_Timing_Schematic.svg`](golden_dataset/AXI_Bus_Timing_Schematic.svg): Vector PCB block schematic and AXI4 burst timing waveform highlighting nets `NET_AXI_AWADDR[6:0]`, `NET_DMA_CR_MBZ_TRAP`, `NET_DMB_SNOOP_INV`, and `NET_SRAM_XBAR_ARB`.
   - [`golden_dataset/oscilloscope_telemetry_dump.csv`](golden_dataset/oscilloscope_telemetry_dump.csv): 2,500-row mixed-signal capture with timestamped anomalies (`t=142.50us` VDD droop to `682.4mV`, `t=418.00us` `56` dirty L1 cache lines, `t=1120.50us` `98.4°C` Bank 0 thermal hotspot).

4. **2D/3D Silicon Die Thermal PDE Simulation ([`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp))**
   - Computes an 80-iteration Gauss-Seidel 2D steady-state heat diffusion PDE ($\nabla^2 T = -P_{\text{density}} / k_{\text{Si}}$) across an `8x8` silicon die mesh during every sandbox execution and renders an interactive **3D Isometric Surface & 2D Floorplan Heatmap** in the browser.

---

## Quick Start & Self-Audit Verification

```powershell
# 1. Run the 4-part Quantitative Self-Audit Suite (64B vs 128B, 8-Bank vs 16-Bank, Compiler Retry, Thermal PDE)
python test_audit.py

# 2. Start the live native g++ sandbox server
python backend/server.py

# 3. Open http://127.0.0.1:8765 in your browser
```
