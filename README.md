# SiliconHeal v3.0: Autonomous `HardwareSpec IR`, Cross-Page Silicon Errata & Proof-Carrying Kernel Self-Healing Agent

> **Track 2 (Hard Tech) · Gemini 4 Argon (`thinking_level="high"`, 1M Multimodal Context) + `antigravity-preview-09-2026` Native C++ Verification Sandbox**
>
> Grounded in 2024–2026 Hardware/Systems AI Research: **SpecMap & Typed HardwareSpec IR (Qiao & Dick 2026)** · **MMLongBench-Doc Cross-Page Override** · **WaveformQA Event-Time Normalization** · **KernelPro / Nsight Semantic Micro-Profiling** · **TherMapNet Causal Thermal/PDN Separation** · **FreeRTOS/QEMU Real-Time WCET & Jitter Verification**

**SiliconHeal v3.0** ingests a multimodal hardware bundle—the **HX-9000 Reference Manual (including Page 88 Section 9 Silicon Errata Rev 4.2)**, **AXI4 Bus Timing & PCB Schematic (`AXI_Bus_Timing_Schematic.svg`)**, and a **2,500-row Mixed-Signal Oscilloscope Telemetry Dump (`oscilloscope_telemetry_dump.csv`)**—alongside a buggy, bottlenecked C++ tensor kernel (`kernel_buggy.cpp`). Instead of translating prose directly to code, SiliconHeal synthesizes a formal **`HardwareSpec IR` (`SpecMap-IR/v3.0`)** with compile-time `static_assert` guards, normalizes raw waveforms into **`WaveformQA` Event-Time JSON**, diagnoses line-level stalls via a **`KernelPro` Semantic Micro-Profiler**, and enforces a **6-Stage Hierarchical Hard Acceptance Gate** (`0/6 -> 1/6 -> 3/6 -> 6/6`) to emit a **Proof-Carrying Patch Evidence Bundle**.

---

## Quantitative Verification Results (Default Spec vs. Mutated Spec)

### 1. Default Spec (`64-Byte` DMA Align · `Bit 7` Reserved · `8-Bank` SRAM)

| Metric | Iter 0 (`kernel_buggy.cpp`) | Iter 1 (DMA Fixed) | Iter 2 (DMB + `volatile`) | Iter 3 (Optimized + `static_assert`) |
| :--- | :--- | :--- | :--- | :--- |
| **Sandbox Verdict** | `HARDWARE_FAULT` (Red) | `DATA_CORRUPTION` (Amber) | `SLA_LATENCY_VIOLATION` | **`ALL_SLA_PASSED` (Green)** |
| **6-Stage Hard Gates** | `0 / 6` Passed | `1 / 6` Passed (`G1`) | `3 / 6` Passed (`G1..G3`) | **`6 / 6` Passed (`G1..G6`)** |
| **DMA Control (`DMA_CR`)** | `0x85` (Bit 7 `RSVD_MBZ` Fault) | `0x05` (Single-Beat) | `0x05` (Single-Beat) | **`0x15` (`BURST_16` Enabled)** |
| **Tensor CRC32 Accuracy** | `0.0%` (Bus Deadlock) | `12.5%` (Stale L1 Cache) | `100.0%` (`0x9DD11DC5`) | **`100.0%` (`0x9DD11DC5`)** |
| **8-Bank SRAM Conflict Rate** | Halted | `94.12%` (8-Way Bank 0 Stall) | `94.12%` (8-Way Bank 0 Stall) | **`0.00%` (`local_tile[8][9]`, 36B stride)** |
| **VDD Core Rail (`di/dt` PDN)** | `682.4 mV` (`18.4 A/ns` spike) | `815.0 mV` (`3.2 A/ns`) | `828.0 mV` (`2.4 A/ns`) | **`842.0 mV` (`1.1 A/ns` clean burst)** |
| **Peak Die Temp (Bank 0 Sense)** | `96.3 °C` (Arbiter Lock) | `97.7 °C` (`45.9 W/mm²`) | `97.7 °C` (`45.9 W/mm²`) | **`62.2 °C` (`7.94 W/mm²` uniform)** |
| **Avg Latency / WCET / Jitter** | Halted (`±9999.0 us`) | `46.51 ms` / `54.88 ms` / `±412.5 us` | `46.51 ms` / `53.02 ms` / `±284.6 us` | **`2.46 ms` (18.91x) / `2.53 ms` / `±4.2 us`** |

### 2. Live Mutated Spec (`128-Byte` DMA Align · `Bit 6` Reserved · `16-Bank` SRAM)

| Verification Scenario | Compile-Time `static_assert` / Declared Align | Tile Stride | Bank Conflict Rate | Avg / WCET Latency | 6-Stage Hard Gates | Sandbox Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old `64B / 8-Bank` Patch on `128B / 16-Bank` HW** | `DMA_BUF_ALIGN=64` vs `HW_DMA_ALIGN_BYTES=128` | `local_tile[8][9]` | Rejected at Compile Time (`static_assert`) | Halted | `0 / 6` | ❌ `COMPILE_STATIC_ASSERT_FAIL` |
| **Live Autonomous Synthesis on `128B / 16-Bank` HW** | **`alignas(128)` + `static_assert` Verified** | **`local_tile[16][17]` (68B)** | **`0.00%`** | **`1.65 ms` / `1.70 ms`** | **`6 / 6`** | ✅ **`ALL_SLA_PASSED`** |

---

## SiliconHeal v3.0 Architecture Upgrades

1. **`HardwareSpec IR` (`SpecMap-IR/v3.0`) & Cross-Page `Silicon Errata Override` ([`backend/multimodal_context.py`](backend/multimodal_context.py) & [`golden_dataset/HX9000_Reference_Manual.md`](golden_dataset/HX9000_Reference_Manual.md))**
   - **Section 9: Silicon Errata Sheet (Rev 4.2, Page 88)** overrides legacy prose in Sections 4.2, 5.1, and 6.4 (`#E-401` AXI4 Burst Alignment & MBZ Lockup, `#E-402` Missing Hardware Cache Snoop & Mandatory `hw_dmb()` + `volatile`, `#E-403` N-Way Wavefront Serialization & Mandatory Coprime `+1` Padding).
   - Synthesizes a formal JSON schema (`registers`, `dma_constraints`, `timing_relations`, `errata_overrides`, `symbol_traceability_map`, `static_assert_guards`) and injects C++17 `static_assert` proofs directly into both the emulator harness and synthesized kernel patches.

2. **`WaveformQA` Event-Time JSON Normalization & `KernelPro` Semantic Micro-Profiler ([`backend/multimodal_context.py`](backend/multimodal_context.py) & [`backend/sandbox_runner.py`](backend/sandbox_runner.py))**
   - Normalizes the 2,500-row oscilloscope CSV into timestamp-aligned **`WaveformQA-EventTime-Telemetry/v3.0`** windows (`EV-AXI-DEADLOCK-01` @ `t=142.50us`, `EV-L1-SNOOP-HAZARD-02` @ `t=418.00us`, `EV-SRAM-WAVEFRONT-HOTSPOT-03` @ `t=1120.50us`) linked to `PC` and schematic nets (`NET_*`).
   - Emits an Nsight Compute / `KernelPro`-style **`semantic_micro_profile`** mapping source C++ line numbers ↔ compiled byte stride ↔ wavefront/bank-conflict serialization ratio ↔ stall root cause.

3. **Physical Causal Graph Separation & Real-Time `wcet_ms` / `jitter_us` Metrics ([`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp))**
   - Strictly decouples **Voltage Droop (`682.4 mV`)** (caused by unaligned DMA trap & `DMA_CR` MBZ bus lock inducing an `18.4 A/ns` PDN $di/dt$ transient spike) from **Thermal Hotspot (`97.7 °C`)** (caused by `32B`/`64B` stride routing all column-gather vector lanes to `Bank 0`, inducing N-way wavefront serialization stalls and `45.9 W/mm²` sense-amp power density).
   - Reports real-time embedded scheduling metrics: **`wcet_ms` (Worst-Case Execution Time)** and **`jitter_us` (Execution Time Jitter)**.

4. **6-Stage Hierarchical Hard Acceptance Gate & `Proof-Carrying Patch Bundle` ([`backend/agent_loop.py`](backend/agent_loop.py) & [`frontend/index.html`](frontend/index.html))**
   - Evaluates 6 ordered hard gates on every iteration: `[G1. Spec Safety (MBZ/Align)] -> [G2. Functional CRC32] -> [G3. Memory/Cache Coherency] -> [G4. Real-Time WCET/Jitter] -> [G5. Throughput/Bank Conflict] -> [G6. Thermal/Voltage Envelope]`.
   - Generates a downloadable/inspectable **`SiliconHeal-ProofCarryingBundle/v3.0`** uniting datasheet/errata bounding boxes, schematic nets, WaveformQA event IDs, C++17 `static_assert` proofs, causal graph separation, and before/after telemetry.

---

## Quick Start & 6-Part Quantitative Self-Audit

```powershell
# 1. Run the 6-part Quantitative Self-Audit Suite (64B vs 128B, 8-Bank vs 16-Bank, Compiler Retry, SpecIR/Errata/WaveformQA/KernelPro, 6-Stage Hard Gates)
python test_audit.py

# 2. Start the live native g++ sandbox server
python backend/server.py

# 3. Open http://127.0.0.1:8765 in your browser
```
