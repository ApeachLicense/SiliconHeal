# SiliconHeal: Autonomous Hardware-Spec & Silicon Errata Kernel Optimization Agent

> **Track 2: Gemini 4 Argon in Hard Tech · `Gemini 4 Argon (thinking_level="high")` + `Antigravity Sandbox (antigravity-preview-09-2026)` + Deterministic Spec Solver Hybrid**
>
> - 💻 **Public GitHub Repository**: **[https://github.com/ApeachLicense/SiliconHeal](https://github.com/ApeachLicense/SiliconHeal)**
> - ☁️ **Cloud-Native Verification**: Containerized with native `g++ -O2 -std=c++17` (`Dockerfile`) for **Google Cloud Run** deployment.

**SiliconHeal** leverages **Gemini 4 Argon (`thinking_level="high"`)**'s 1M-token unified multimodal context and **Antigravity Sandbox (`antigravity-preview-09-2026`)** verification loops to ingest a heterogeneous hardware bundle—the **HX-9000 Reference Manual (including Page 88 Section 9 Silicon Errata Rev 4.2)**, **AXI4 Bus Timing & PCB Schematic (`AXI_Bus_Timing_Schematic.svg`)**, and a **2,500-row Mixed-Signal Oscilloscope Telemetry Dump (`oscilloscope_telemetry_dump.csv`)**—alongside a buggy, bottlenecked C++ tensor kernel (`kernel_buggy.cpp`).

Instead of relying on kernel self-reporting or hardcoded patch files, SiliconHeal combines **Gemini 4 Argon (`thinking_level="high") + Deterministic Spec Solver Hybrid** reasoning with an instrumented C++17 verification sandbox:
1. Parses hardware constraints (bit positions, bank modulo formulas, and cross-page errata overrides) into a **`HardwareSpec IR`** (`SiliconHeal-HardwareSpecIR/v3.1`) with compile-time `static_assert` guards.
2. Scans the 2,500-row oscilloscope CSV by signal thresholds (`axi_wvalid==1 & axi_wready==0`, `l1_dirty_lines>0`, `bank_conflict_ways>1`) to extract timestamp-aligned anomaly windows.
3. Compiles and runs every synthesized kernel in an **Antigravity Sandbox (`antigravity-preview-09-2026`)** powered by native `g++ -O2 -std=c++17` (`LC_ALL=C` cross-platform Linux/Windows compatible) where all on-chip scratchpad traffic is intercepted at the harness boundary by an instrumented C++ proxy template **`SramTile<T, Rows, Cols>`**.
4. Enforces a **6-Stage Hierarchical Acceptance Gate** (`0/6 -> 1/6 -> 3/6 -> 6/6`), rejecting any kernel that bypasses `SramTile` (`ERR_SRAM_TILE_BYPASS_DETECTED`).

---

## Quantitative Verification Results (Mechanistic Runtime Counters)

### 1. Default Spec (`64-Byte` DMA Align · `Bit 7` Reserved · `8-Bank` SRAM)

| Metric | Iter 0 (`kernel_buggy.cpp`) | Iter 1 (DMA Fixed) | Iter 2 (DMB + `volatile`) | Iter 3 (`SramTile<float, 8, 9>` + `BURST_16`) |
| :--- | :--- | :--- | :--- | :--- |
| **Sandbox Verdict** | `HARDWARE_FAULT` (Red) | `DATA_CORRUPTION` (Amber) | `SLA_LATENCY_VIOLATION` | **`ALL_SLA_PASSED` (Green)** |
| **6-Stage Gates** | `0 / 6` Passed | `1 / 6` Passed (`G1`) | `3 / 6` Passed (`G1..G3`) | **`6 / 6` Passed (`G1..G6`)** |
| **DMA Control (`DMA_CR`)** | `0x85` (Bit 7 `RSVD_MBZ` Fault) | `0x05` (Single-Beat) | `0x05` (Single-Beat) | **`0x15` (`BURST_16` Synthesized from Bits 4,2,0)** |
| **`SramTile` Word Reads / Writes** | `0 / 0` (Aborted at DMA) | `65,536 / 65,536` | `65,536 / 65,536` | **`65,536 / 65,536` (100% Instrumented)** |
| **Tensor CRC32 Accuracy** | `0.0%` (Bus Deadlock) | `12.5%` (Stale L1 Cache) | `100.0%` (`0x9DD11DC5`) | **`100.0%` (`0x9DD11DC5`)** |
| **8-Bank SRAM Conflict Rate** | Halted | `87.50%` (7/8 Colliding Lanes) | `87.50%` (7/8 Colliding Lanes) | **`0.00%` (`gcd(9, 8) == 1`, 36B Stride)** |
| **Avg Latency / WCET / Jitter** | Halted (`±9999.0 us`) | `42.76 ms` / `45.41 ms` / `±365.0 us` | `42.76 ms` / `45.36 ms` / `±335.0 us` | **`2.58 ms` (16.57x Speedup) / `2.65 ms` / `±4.2 us`** |
| **Peak Die Temp / VDD Core** | `97.7 °C` / `682.4 mV` | `100.0 °C` / `818.0 mV` | `100.0 °C` / `830.0 mV` | **`62.2 °C` / `844.5 mV`** |

### 2. Live Mutated Spec (`128-Byte` DMA Align · `Bit 6` Reserved · `16-Bank` SRAM)

| Verification Scenario | Compile-Time `static_assert` / Declared Align | Scratchpad Container | Bank Conflict Rate | Avg / WCET Latency | 6-Stage Gates | Sandbox Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Old `64B / 8-Bank` Patch on `128B / 16-Bank` HW** | `DMA_BUF_ALIGN=64` vs `HW_DMA_ALIGN_BYTES=128` | `SramTile<float, 8, 9>` | Rejected at Compile Time (`static_assert`) | Halted | `0 / 6` | ❌ `COMPILE_STATIC_ASSERT_FAIL` |
| **Unpadded `16×16` Tile on `16-Bank` HW (Iter 1–2)** | `alignas(128)` | `SramTile<float, 16, 16>` (64B) | `93.75%` (15/16 Lanes Collide) | `28.02 ms` / `29.75 ms` | `3 / 6` | ❌ `SLA_LATENCY_VIOLATION` |
| **Synthesized Coprime Patch on `16-Bank` HW (Iter 3)** | **`alignas(128)` + `static_assert` Verified** | **`SramTile<float, 16, 17>` (68B)** | **`0.00%` (`gcd(17, 16) == 1`)** | **`2.21 ms` / `2.28 ms`** | **`6 / 6`** | ✅ **`ALL_SLA_PASSED`** |

### 3. Anti-Cheat Verification (`AUDIT 3B` in `test_audit.py`)

| Adversarial Kernel Attempt | Observed `SramTile` Reads/Writes | Measured Bank Conflict | Measured Latency | 6-Stage Gates | Sandbox Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cheat Kernel A**: Raw `float local_tile[8][8]` (bypasses `SramTile` to fake `0%` conflicts) | `0 / 0` | `0.00%` | Aborted | `0 / 6` | ❌ `HARDWARE_FAULT` (`ERR_SRAM_TILE_BYPASS_DETECTED`) |
| **Cheat Kernel B**: Unpadded `SramTile<float, 8, 8>` + `BURST16` with zero manual reporting calls | `65,536 / 65,536` | **`87.50%`** (`57,344 / 65,536`) | **`33.55 ms`** (`> 3.50ms` SLA) | `3 / 6` | ❌ `SLA_LATENCY_VIOLATION` |

---

## Key Engineering Architecture

1. **Harness-Enforced `SramTile<T, Rows, Cols>` Instrumentation ([`golden_dataset/hw_emulator_harness.hpp`](golden_dataset/hw_emulator_harness.hpp) & [`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp))**
   - Kernels do **not** self-report bank conflicts. Instead, `SramTile<T, Rows, Cols>` wraps the scratchpad buffer and intercepts every element read (`operator const T&`) and write (`operator=`) via `RowProxy` and `ElemProxy`.
   - When the inner transpose loop reads `local_tile[r][c]` across rows `r = 0..TILE_DIM-1` with fixed column `c`, the proxy detects the column-major stride (`Cols * sizeof(T)`) and evaluates bank collisions across all `NUM_BANKS` parallel lanes (`bank = ((lane * Cols) + c) % NUM_BANKS`).
   - Before checking CRC32 or latency, `run_verification_harness()` verifies that `sram_word_reads >= 65536`, `sram_word_writes >= 65536`, and `strided_transpose_groups >= 65536 / TILE_DIM`. Any raw-array bypass immediately fails with `ERR_SRAM_TILE_BYPASS_DETECTED`.

2. **Genuine Cross-Page Silicon Errata Contradiction & Zero-Leak Constraint Derivation ([`golden_dataset/HX9000_Reference_Manual.md`](golden_dataset/HX9000_Reference_Manual.md) & [`backend/multimodal_context.py`](backend/multimodal_context.py))**
   - **Cross-Page Override (`#E-402`)**: Section 5.1 (Page 27) states that hardware automatically snoops L1 D-Cache lines during `BURST_16` transfers. Section 9 Errata `#E-402` (Page 88) overrides Section 5.1 for Rev 4.2 silicon, stating the hardware snoop filter is disabled due to an AXI4 `ARPROT` comparator defect and mandating explicit `hw_dmb()` and `hw_poll_status_volatile()`.
   - **Zero Answer Leaks**: Neither `0x15` nor `TILE_DIM + 1` nor `PADDED_STRIDE` appears in the datasheet prose. The parser extracts bit positions (`Bit 4`, `Bit 2`, `Bit 0`, forbidden `Bit 7`/`Bit 6`) from the Section 4.2.2 register table to compute `0x15`, and solves `math.gcd(cols, num_banks) == 1` to derive the coprime column width (`9` for 8-bank, `17` for 16-bank).

3. **Mechanistic Cycle Equations & Explicit Telemetry Provenance ([`golden_dataset/hw_emulator_harness.cpp`](golden_dataset/hw_emulator_harness.cpp))**
   - **Cycle Model**: `dma_cycles` is computed from 64-bit AXI4 beats (`beats / 16 * 18` cycles for `BURST_16` vs. `beats * 26` cycles for single-beat); `sram_compute_cycles` is computed from `sram_word_reads`, `sram_word_writes`, and `total_bank_conflicts * 10` crossbar arbitration penalty cycles (`latency_ms = total_cycles / 2500.0`).
   - **Telemetry Provenance**: The emulator output includes a `telemetry_provenance` block explicitly documenting which fields are `runtime_hardware_counters` (`sram_word_reads`, `sram_word_writes`, `bank_access_hist`, `bank_conflict_hist`, `crc_actual`, `total_cycles`, `latency_ms`) and which are `counter_driven_analytical_models` (`die_thermal_grid_c` 2D finite-difference diffusion driven by `bank_conflict_hist[]`, `vdd_droop_mv`, `wcet_ms`, `jitter_us`).

4. **`Gemini 4 Argon (thinking_level="high")` + Deterministic Spec Solver Hybrid & Cross-Platform `g++` Self-Healing ([`backend/agent_loop.py`](backend/agent_loop.py) & [`backend/sandbox_runner.py`](backend/sandbox_runner.py))**
   - Subprocess `g++` calls inside the **Antigravity Sandbox (`antigravity-preview-09-2026`)** enforce `LC_ALL=C` and `LANG=C`, and `repair_cpp_syntax_errors()` normalizes Unicode curly quotes (`‘’“”` -> `'`) before parsing `g++` `stderr` diagnostics, ensuring identical behavior on Linux and Windows.
   - Supports **Gemini 4 Argon (`gemini-4-argon`, `thinking_level="high"`)** via Vertex AI / Google AI Studio (`GEMINI_API_KEY`) paired with a deterministic `HardwareSpec IR` constraint solver (`Gemini 4 Argon (thinking_level="high") + Deterministic Spec Solver Hybrid`).

---

## Quick Start, Self-Audit & Google Cloud Run Deployment

```bash
# 1. Run the 7-part Quantitative & Anti-Cheat Self-Audit Suite (Linux & Windows g++ compatible)
python test_audit.py

# 2. Start the HTTP server (binds 0.0.0.0:$PORT, default port 8765)
python backend/server.py

# 3. Deploy to Google Cloud Run (replace $GCP_PROJECT_ID with your project ID)
gcloud run deploy siliconheal --source . --project $GCP_PROJECT_ID --region us-central1 --allow-unauthenticated
```
