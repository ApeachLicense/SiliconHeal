# HX-9000 Edge AI Tensor & DMA Coprocessor Reference Manual
**Document ID:** HX9000-RM-REV4.2  
**Architecture:** ARMv8-M / Custom AXI4-Stream Neural Tile Engine  
**Clock Domain:** 400 MHz Core / 200 MHz AXI4 Bus  

---

## Section 3. Memory Map & MMIO Register Summary (Page 9)

The HX-9000 DMA and Local SRAM Controller are mapped to the peripheral non-cacheable address space starting at base address `0x4002_1000`.

| Offset | Register Name | Access | Reset Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `DMA_CR` | R/W | `0x0000_0000` | DMA Control & Burst Configuration Register |
| `0x04` | `DMA_SR` | RO (Volatile) | `0x0000_0000` | DMA Status & Hardware Fault Code Register |
| `0x08` | `DMA_SRC_ADDR` | R/W | `0x0000_0000` | System DRAM Source Physical Address |
| `0x0C` | `DMA_DST_ADDR` | R/W | `0x2000_0000` | Local TCM SRAM Destination Address |
| `0x10` | `DMA_LEN` | R/W | `0x0000_0000` | Transfer Length in Bytes (Multiple of 64) |

---

## Section 4.2 DMA Alignment & Control Register Constraints (Page 14)

### 4.2.1 64-Byte Cache-Line Address Alignment Rule [CRITICAL]
Both `DMA_SRC_ADDR` and `DMA_DST_ADDR` **MUST be aligned to a 64-byte boundary** (`address % 64 == 0`, i.e., `address & 0x3F == 0`).
* **Hardware Fault Behavior:** Initiating a DMA transfer with an unaligned source or destination pointer immediately raises `ERR_DMA_UNALIGNED_64B` (`DMA_SR[2] = 1`) and aborts the kernel execution.
* **C/C++ Implementation Requirement:** Buffers passed to the DMA engine must be allocated or declared with `alignas(64)`.

### 4.2.2 `DMA_CR` (Offset `0x00`) Bitfield Specification
| Bits | Name | Description |
| :--- | :--- | :--- |
| `31:8` | `RESERVED` | Reserved. Read as 0, must be written as 0. |
| **`7`** | **`RSVD_MBZ`** | **Reserved Must-Be-Zero (Factory Debug Bus Lock).**<br>**WARNING:** Writing `1` to Bit 7 (`0x80`) locks the AXI4 arbiter and triggers an immediate `ERR_AXI_BUS_DEADLOCK` (`DMA_SR[3] = 1`) hardware fault. |
| `6:5` | `PRIORITY` | Channel priority (`00` = Low, `01` = Medium, `10` = High). |
| **`4`** | **`BURST_16`** | **16-Beat AXI4 Burst Mode Enable.**<br>`0` = Single-beat transfer (high bus overhead).<br>`1` = 16-beat burst mode (`0x10`), reducing DMA transfer overhead by 8.2x. |
| `2` | `DIR_MEM2SRAM`| `1` = DRAM to Local SRAM transfer (`0x04`). |
| `0` | `EN` | `1` = Enable and trigger DMA transfer (`0x01`). |

> **Valid Configuration Example:**  
> Standard single-beat enable: `DMA_CR = 0x05` (`DIR_MEM2SRAM | EN`).  
> Optimized 16-beat burst enable: `DMA_CR = 0x15` (`BURST_16 | DIR_MEM2SRAM | EN`).  
> **INVALID:** `DMA_CR = 0x85` sets Bit 7 (`RSVD_MBZ`) and causes an immediate hardware fault!

---

## Section 5.1 Cache Coherency & MMIO Polling Protocol (Page 27)

### 5.1.1 Mandatory Data Memory Barrier (`DMB`) Before DMA Trigger
The HX-9000 DMA engine reads directly from physical DRAM, bypassing the CPU's L1 Data Cache.
* If the CPU writes or modifies input tensor data prior to setting `DMA_CR.EN`, the dirty cache lines remain in L1 cache unless explicitly flushed/ordered.
* **Hardware Rule:** Software **MUST invoke `hw_dmb()` (Data Memory Barrier / Cache CleanFence) immediately before writing to `DMA_CR`**. Omitting `hw_dmb()` results in the DMA engine reading stale/uninitialized memory, causing silent **Output CRC32 Data Corruption**.

### 5.1.2 `volatile` Qualification on `DMA_SR` Polling
The status register `DMA_SR` (Offset `0x04`) is updated asynchronously by the DMA state machine (`Bit 0: DONE`, `Bit 1: BUSY`).
* Polling `DMA_SR` without `volatile` semantics (`hw_read_status_volatile()`) allows optimizing compilers (`-O2`/`-O3`) to hoist the read out of the loop or read a cached register state before transfer completion, leading to corrupted partial tile reads.

---

## Section 6.4 8-Bank Interleaved Local SRAM & Bank Conflict Avoidance (Page 41)

### 6.4.1 8-Bank Interleaved Memory Architecture
The 64 KB Local Tightly-Coupled SRAM is partitioned into **8 parallel 32-bit (4-byte) banks** (`Bank 0` through `Bank 7`).
For any byte offset `addr`, the hardware bank index is determined by:
$$\text{Bank Index} = \left\lfloor \frac{\text{addr}}{4} \right\rfloor \bmod 8$$

Consecutive 32-bit floats (`4 bytes` apart) map to consecutive banks (`0, 1, 2, 3, 4, 5, 6, 7`), enabling 8 parallel accesses per cycle when each lane hits a distinct bank.

### 6.4.2 The 32-Byte Stride Bank Conflict Trap
When executing a $8 \times 8$ (or $N \times 8$ tile) matrix transpose or column-major reduction on `float` (`4 bytes`):
* Row stride is $8 \times 4\text{ bytes} = 32\text{ bytes}$.
* Since $32 / 4 = 8$, accessing column elements `tile[r][c]` across rows `r = 0..7` addresses byte offsets `0, 32, 64, 96, ...`, which **ALL map to the exact same SRAM bank**:
  $$\left(\frac{32 \cdot r + 4 \cdot c}{4}\right) \bmod 8 = (8r + c) \bmod 8 = c$$
* **Performance Impact:** All 8 parallel lanes collide on Bank $c$ (**94.12% Bank Conflict Rate**), serializing memory access and stalling the pipeline by **12 cycles per conflict way**.

### 6.4.3 Recommended Hardware Optimization: +1 Element Row Padding
To achieve **0% Bank Conflict** during column-wise or transpose tile access:
1. Pad the local SRAM tile declaration by **+1 float per row**: `float local_tile[TILE_DIM][TILE_DIM + 1]` (stride becomes $9 \times 4 = 36\text{ bytes}$).
2. Because $\gcd(9, 8) = 1$, consecutive rows `r = 0..7` in the same column `c` map to banks `(9r + c) mod 8`, hitting all 8 banks without a single collision!
3. Combined with `DMA_CR = 0x15` (`BURST_16`), kernel latency improves from **46.51 ms to 2.46 ms (18.91x speedup)**.

---

## Section 9. Silicon Errata Sheet — Rev 4.2 Mandatory Cross-Page Overrides (Page 88)

> **CRITICAL SILICON ADVISORY (Cross-Page Specification Override):**  
> In Rev 4.2 silicon steppings, three hardware errata (`#E-401`, `#E-402`, `#E-403`) **override and supersede** the nominal programming guidelines in Sections 4.2, 5.1, and 6.4. Any firmware or kernel synthesis agent MUST apply the following workarounds and compile-time `static_assert` proofs.

| Errata ID | Overrides Section | Hardware Defect & Electrical/Thermal Root Cause | Mandatory Software Workaround & `static_assert` Guard |
| :--- | :--- | :--- | :--- |
| **`#E-401`** | **Overrides Sec 4.2.1 & 4.2.2 (Page 14)** | **AXI4 Arbiter Latch-Up & PDN $di/dt$ Voltage Droop:** Unaligned `DMA_DST_ADDR` or asserting `DMA_CR[RSVD_MBZ]` triggers an unrecoverable arbiter retry oscillation on net `NET_DMA_CR_MBZ_TRAP`, inducing a `18.4 A/ns` $di/dt$ spike across package inductance and dropping `VDD_CORE` to `682.4 mV` at `t=142.50us`. | Enforce `alignas(HW_DMA_ALIGN_BYTES)` at base offset `[0]`, clear `DMA_CR_RSVD_MBZ`, enable `DMA_CR_BURST16` (`0x15`), and prove alignment via `static_assert`. |
| **`#E-402`** | **Overrides Sec 5.1.1 & 5.1.2 (Page 27)** | **L1 D-Cache Snoop Filter Bypass Hazard:** Hardware snoop controller on net `NET_DMB_SNOOP_INV` fails to auto-clean dirty CPU L1 lines (`56 lines` at `t=418.00us`) during non-coherent DMA reads, while `-O2` loop hoisting caches stale `DMA_SR` status bits. | Execute explicit `hw_dmb()` immediately before `hw_trigger_dma()` AND poll completion exclusively via `hw_poll_status_volatile()`. |
| **`#E-403`** | **Overrides Sec 6.4.1 & 6.4.2 (Page 41)** | **Crossbar Wavefront Serialization & Bank 0 Thermal Runaway:** Power-of-two tile row stride (`TILE_DIM` floats) causes 100% column-gather serialization on `NET_SRAM_XBAR_ARB` (`Bank 0`), elevating local sense-amplifier power density to `45.9 W/mm²` (`97.7°C ~ 99.2°C` hotspot at `t=1120.50us`) and violating WCET/Jitter bounds. | Override nominal `TILE_DIM` stride with coprime `PADDED_STRIDE = TILE_DIM + 1` and enforce `static_assert((PADDED_STRIDE % TILE_DIM) == 1, "Errata #E-403")`. |
