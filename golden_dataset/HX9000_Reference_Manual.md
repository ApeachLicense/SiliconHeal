# HX-9000 Edge AI Tensor & DMA Coprocessor Reference Manual
**Document ID:** HX9000-RM-REV4.2  
**Architecture:** ARMv8-M / Custom AXI4-Stream Neural Tile Engine  
**Clock Domain:** 400 MHz Core / 200 MHz AXI4 Bus (64-bit Data Width)  

---

## Section 3. Memory Map & MMIO Register Summary (Page 9)

The HX-9000 DMA and Local SRAM Controller are mapped to the peripheral address space starting at base address `0x4002_1000`.

| Offset | Register Name | Access | Reset Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `DMA_CR` | R/W | `0x0000_0000` | DMA Control & Burst Configuration Register |
| `0x04` | `DMA_SR` | RO | `0x0000_0000` | DMA Status & Hardware Fault Code Register |
| `0x08` | `DMA_SRC_ADDR` | R/W | `0x0000_0000` | System DRAM Source Physical Address |
| `0x0C` | `DMA_DST_ADDR` | R/W | `0x2000_0000` | Local Staging Destination Physical Address |
| `0x10` | `DMA_LEN` | R/W | `0x0000_0000` | Transfer Length in Bytes |

---

## Section 4.2 DMA Alignment & Control Register Specification (Page 14)

### 4.2.1 DMA Address Alignment Specification
Both `DMA_SRC_ADDR` and `DMA_DST_ADDR` must be aligned to a **64-byte boundary** (`address % 64 == 0`).
* **Fault Behavior:** Initiating a DMA transfer with an unaligned source or destination address raises `ERR_DMA_UNALIGNED` (`DMA_SR[2] = 1`) and aborts transfer execution.

### 4.2.2 `DMA_CR` (Offset `0x00`) Bitfield Table
Software constructs the `DMA_CR` control word by combining the required bit positions below:

| Bit Position | Field Name | Value & Hardware Function |
| :--- | :--- | :--- |
| `31:8` | `RESERVED` | Reserved. Must be written as `0`. |
| **`7`** | **`RSVD_MBZ`** | **Reserved Must-Be-Zero (Internal Test Mode).** See Section 9 Errata `#E-401`. |
| `6:5` | `PRIORITY` | Channel arbitration priority (`00` = Standard). |
| **`4`** | **`BURST_16`** | **AXI4 Burst Length Select.**<br>`0` = Single-beat mode (1 beat per bus grant; 13 core cycles arbitration overhead per beat).<br>`1` = 16-beat pipelined burst mode (16 beats per bus grant; 8 core cycles setup per burst). |
| `3` | `CIRC_MODE` | `0` = Linear one-shot transfer; `1` = Circular ring buffer. |
| **`2`** | **`DIR_MEM2SRAM`**| **Transfer Direction.**<br>`1` = System DRAM to Local Staging Buffer.<br>`0` = Local Staging Buffer to System DRAM. |
| `1` | `IRQ_EN` | `0` = Disable completion IRQ (use status polling); `1` = Enable IRQ. |
| **`0`** | **`EN`** | **Channel Enable.**<br>`1` = Initiate DMA transfer; `0` = Channel disabled. |

---

## Section 5.1 Nominal Cache Coherency & Status Polling (Page 27)

> **NOTE:** See **Section 9 (Silicon Errata `#E-402`, Page 88)** for critical Rev 4.2 silicon limitations that override this section.

### 5.1.1 Hardware Cache Snooping (Nominal Specification)
In the nominal HX-9000 architecture, the AXI4 DMA master interfaces with an L1 Data Cache snoop filter designed to maintain coherency during memory-to-scratchpad transfers.
* **Explicit Barrier API:** The HAL provides `hw_dmb()` to force an explicit Data Memory Barrier and L1 cache clean prior to `hw_trigger_dma(...)`.

### 5.1.2 Status Register (`DMA_SR`) Polling APIs
The HAL provides two status read helpers for `DMA_SR` (`Bit 0: DONE`):
* `hw_poll_status_cached()`: Standard non-volatile register read.
* `hw_poll_status_volatile()`: Strictly ordered `volatile` MMIO register read.

---

## Section 6.4 Interleaved Local SRAM Architecture (Page 41)

### 6.4.1 8-Bank Word-Interleaved Scratchpad
The Local Tightly-Coupled Scratchpad SRAM (`SramTile<float, Rows, Cols>`) is partitioned into **8 parallel 32-bit (4-byte word) banks** (`Bank 0` through `Bank 7`).
For any 2D tile element `(r, c)` stored in a tile container with column dimension `Cols`, its linear word offset is `w = r * Cols + c`, and its hardware bank index is:
$$\text{Bank\_ID}(r, c) = (r \cdot \text{Cols} + c) \bmod 8$$

### 6.4.2 Vector Crossbar Access Pattern
During an 8-lane vector operation:
* **Row-Wise Access (`r` fixed, `c = 0..7`):** Word offsets increment by `1`, mapping lanes to `(r * Cols + 0..7) mod 8`, which accesses all 8 banks in parallel (1 wavefront cycle).
* **Column-Wise Gather (`c` fixed, `r = 0..7`):** Word offsets increment by `Cols`. The bank accessed by lane `r` is `(r * Cols + c) mod 8`.

---

## Section 9. Silicon Errata Sheet — Rev 4.2 Mandatory Cross-Page Overrides (Page 88)

> **CRITICAL SILICON ADVISORY (Cross-Page Specification Override):**  
> In Rev 4.2 silicon steppings, three hardware defects (`#E-401`, `#E-402`, `#E-403`) **override and supersede** the nominal behaviors described in Sections 4.2, 5.1, and 6.4.

| Errata ID | Overrides Section | Silicon Defect & Observed Symptom | Architectural Constraint Required |
| :--- | :--- | :--- | :--- |
| **`#E-401`** | **Overrides Sec 4.2 (Page 14)** | **AXI4 Write-Response State Machine Lockup:** Asserting `DMA_CR` Reserved Bit `7` (`RSVD_MBZ`) or supplying an address not aligned to `64` bytes locks the AXI4 arbiter (`AWVALID=1, AWREADY=0`), causing high-frequency retry current spikes on `VDD_CORE`. Additionally, leaving `BURST_16=0` degrades DMA throughput by `6x` due to per-beat bus re-arbitration. | The staging buffer pointer must start at a `64`-byte aligned base address, `DMA_CR` must have `RSVD_MBZ` cleared (`0`), and `EN`, `DIR_MEM2SRAM`, and `BURST_16` must be asserted (`1`) for production throughput. |
| **`#E-402`** | **Overrides Sec 5.1 (Page 27)** | **Hardware L1 Snoop Filter Disabled in Rev 4.2:** Contrary to Section 5.1.1, the automatic hardware snoop filter is non-functional in Rev 4.2 silicon due to a clock-domain crossing race on `NET_DMB_SNOOP_INV`. Starting DMA without a manual barrier reads stale DRAM lines, and non-volatile status polling (`hw_poll_status_cached()`) returns stale completion state under `-O2`. | Software **must ignore** the nominal auto-snoop assumption in Section 5.1: explicitly call `hw_dmb()` immediately before `hw_trigger_dma()`, and replace `hw_poll_status_cached()` with `hw_poll_status_volatile()`. |
| **`#E-403`** | **Overrides Sec 6.4 (Page 41)** | **Crossbar Bank-Conflict Serialization & Sense-Amp Heating:** When `SramTile<float, Rows, Cols>` uses a column width `Cols` that shares a common factor with the bank count `N_BANKS = 8` ($\gcd(\text{Cols}, 8) > 1$), column-wise gather (`r = 0..7`) maps multiple lanes to the same bank, stalling the crossbar and concentrating switching power in `Bank 0`. | The second template dimension `Cols` of `SramTile<float, Rows, Cols>` must satisfy $\text{Cols} \ge \text{TILE\_DIM}$ and $\gcd(\text{Cols}, 8) = 1$ (coprime to the number of SRAM banks) so that `(r * Cols + c) mod 8` is a bijection across all 8 banks. |
