#ifndef HW_EMULATOR_HARNESS_HPP
#define HW_EMULATOR_HARNESS_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// HX-9000 Cycle-Accurate Hardware Register & 8-Bank SRAM Sandbox Emulator
// ============================================================================

static constexpr int MATRIX_DIM = 256;          // 256 x 256 matrix (65,536 floats = 256 KB)
static constexpr int TILE_DIM   = 8;            // 8 x 8 hardware vector tile
static constexpr int NUM_BANKS  = 8;            // 8-Bank Interleaved Local SRAM

// DMA_CR Bitmasks (Manual Section 4.2.2)
static constexpr uint32_t DMA_CR_EN          = 0x01; // Bit 0: Enable
static constexpr uint32_t DMA_CR_DIR_M2S     = 0x04; // Bit 2: DRAM -> SRAM
static constexpr uint32_t DMA_CR_BURST16     = 0x10; // Bit 4: 16-Beat Burst
static constexpr uint32_t DMA_CR_RSVD_MBZ    = 0x80; // Bit 7: Reserved Must-Be-Zero (FAULT!)

// DMA_SR Fault & Status Bits
static constexpr uint32_t DMA_SR_DONE             = 0x01;
static constexpr uint32_t DMA_SR_ERR_UNALIGNED_64 = 0x04; // Bit 2
static constexpr uint32_t DMA_SR_ERR_AXI_DEADLOCK = 0x08; // Bit 3

struct HX9000_State {
    uint32_t DMA_CR       = 0;
    volatile uint32_t DMA_SR = 0;
    uintptr_t DMA_SRC_ADDR = 0;
    uintptr_t DMA_DST_ADDR = 0;
    uint32_t DMA_LEN      = 0;

    bool dmb_called_before_dma = false;
    bool volatile_status_polled = false;
    bool fault_triggered       = false;
    char fault_code[128]       = "NONE";
    char fault_detail[256]     = "No hardware fault.";

    // Telemetry Counters
    uint64_t total_cycles          = 0;
    uint64_t dma_cycles            = 0;
    uint64_t sram_compute_cycles   = 0;
    uint64_t total_sram_groups     = 0;
    uint64_t conflicted_sram_groups = 0;
    uint64_t bank_access_hist[NUM_BANKS]    = {0};
    uint64_t bank_conflict_hist[NUM_BANKS]  = {0};
};

inline HX9000_State g_hw;

// Call before CPU writes to DRAM to mark cache dirty
inline void hw_mark_cache_dirty() {
    g_hw.dmb_called_before_dma = false;
}

// Section 5.1.1: Mandatory Data Memory Barrier before DMA trigger
inline void hw_dmb() {
    g_hw.dmb_called_before_dma = true;
    g_hw.total_cycles += 4;
}

// Non-volatile status poll (BUGGY: violates Section 5.1.2)
inline uint32_t hw_poll_status_cached() {
    g_hw.volatile_status_polled = false;
    return 0x01; // Compiler cached value before DMA completion
}

// Volatile status poll (COMPLIANT: Section 5.1.2)
inline uint32_t hw_poll_status_volatile() {
    g_hw.volatile_status_polled = true;
    g_hw.total_cycles += 2;
    return g_hw.DMA_SR;
}

// Trigger DMA engine and validate Section 4.2 & Section 5.1 hardware constraints
inline bool hw_trigger_dma(const float* src_ptr, float* dst_ptr, uint32_t bytes, uint32_t dma_cr_val) {
    g_hw.DMA_SRC_ADDR = reinterpret_cast<uintptr_t>(src_ptr);
    g_hw.DMA_DST_ADDR = reinterpret_cast<uintptr_t>(dst_ptr);
    g_hw.DMA_LEN      = bytes;
    g_hw.DMA_CR       = dma_cr_val;

    // 1. Check Section 4.2.2: Bit 7 Reserved Must-Be-Zero
    if (dma_cr_val & DMA_CR_RSVD_MBZ) {
        g_hw.DMA_SR |= DMA_SR_ERR_AXI_DEADLOCK;
        g_hw.fault_triggered = true;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_AXI_BUS_DEADLOCK");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 4.2.2 Violation: DMA_CR (0x%02X) sets Reserved Bit 7 (0x80 RSVD_MBZ), locking AXI4 arbiter.",
            dma_cr_val);
        return false;
    }

    // 2. Check Section 4.2.1: 64-Byte Alignment on Source & Destination
    if ((g_hw.DMA_SRC_ADDR & 0x3FULL) != 0 || (g_hw.DMA_DST_ADDR & 0x3FULL) != 0) {
        g_hw.DMA_SR |= DMA_SR_ERR_UNALIGNED_64;
        g_hw.fault_triggered = true;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_DMA_UNALIGNED_64B");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 4.2.1 Violation: DMA address (SRC=0x%llX, DST=0x%llX) is not 64-byte aligned (addr & 0x3F != 0).",
            (unsigned long long)g_hw.DMA_SRC_ADDR, (unsigned long long)g_hw.DMA_DST_ADDR);
        return false;
    }

    // 3. Calculate DMA Bus Cycles (Section 4.2.2: BURST_16 vs Single-Beat)
    uint64_t beats = bytes / 16;
    uint64_t step_dma_cycles = (dma_cr_val & DMA_CR_BURST16) ? (beats * 3ULL) : (beats * 42ULL);
    g_hw.dma_cycles += step_dma_cycles;
    g_hw.total_cycles += step_dma_cycles;

    // 4. Perform transfer (Simulate Section 5.1.1 Stale L1 Cache if hw_dmb() omitted)
    const size_t count = bytes / sizeof(float);
    if (g_hw.dmb_called_before_dma) {
        std::memcpy(dst_ptr, src_ptr, bytes);
    } else {
        // Stale cache line corruption: only every 8th element was flushed to DRAM
        for (size_t i = 0; i < count; ++i) {
            dst_ptr[i] = (i % 8 == 0) ? src_ptr[i] : -999.0f;
        }
    }

    g_hw.DMA_SR = DMA_SR_DONE;
    return true;
}

// Simulate 8-Lane Vector Load/Store on 8-Bank Interleaved Local SRAM (Section 6.4)
// Accesses 8 elements at byte offsets base_floats[0..7] * 4
inline void hw_record_8lane_sram_access(const int elem_indices[8]) {
    int bank_hits[NUM_BANKS] = {0};
    int max_way_conflict = 1;

    for (int lane = 0; lane < 8; ++lane) {
        int bank = ((elem_indices[lane]) % NUM_BANKS + NUM_BANKS) % NUM_BANKS;
        bank_hits[bank]++;
        g_hw.bank_access_hist[bank]++;
        if (bank_hits[bank] > max_way_conflict) {
            max_way_conflict = bank_hits[bank];
        }
    }

    g_hw.total_sram_groups++;
    if (max_way_conflict > 1) {
        g_hw.conflicted_sram_groups++;
        for (int b = 0; b < NUM_BANKS; ++b) {
            if (bank_hits[b] > 1) {
                g_hw.bank_conflict_hist[b] += (bank_hits[b] - 1);
            }
        }
        // Section 6.4.2: Bank conflict serializes pipeline with 16x stall penalty per collision way
        uint64_t penalty = static_cast<uint64_t>(max_way_conflict) * 32ULL;
        g_hw.sram_compute_cycles += penalty;
        g_hw.total_cycles += penalty;
    } else {
        // Conflict-free parallel 8-bank access: 1 cycle!
        g_hw.sram_compute_cycles += 1;
        g_hw.total_cycles += 1;
    }
}

#endif // HW_EMULATOR_HARNESS_HPP
