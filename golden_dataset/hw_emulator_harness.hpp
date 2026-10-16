#ifndef HW_EMULATOR_HARNESS_HPP
#define HW_EMULATOR_HARNESS_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// HX-9000 Cycle-Accurate & Thermal-Aware Hardware Sandbox Emulator (Rev 4.2+)
// Supports Live Spec Mutation via Compile-Time Macros (-DHW_*)
// ============================================================================

#ifndef HW_DMA_ALIGN_BYTES
#define HW_DMA_ALIGN_BYTES 64
#endif

#ifndef HW_RSVD_MBZ_BIT
#define HW_RSVD_MBZ_BIT 7
#endif

#ifndef HW_NUM_BANKS
#define HW_NUM_BANKS 8
#endif

#ifndef HW_SLA_TARGET_MS
#define HW_SLA_TARGET_MS 3.50
#endif

#ifndef HW_KERNEL_DECLARED_ALIGN
#define HW_KERNEL_DECLARED_ALIGN 64
#endif

static constexpr int MATRIX_DIM = 256;                 // 256 x 256 tensor (65,536 floats = 256 KB)
static constexpr int TILE_DIM   = HW_NUM_BANKS;        // Vector tile matches bank width (8 or 16)
static constexpr int NUM_BANKS  = HW_NUM_BANKS;        // 8-Bank or 16-Bank Interleaved Local SRAM
static constexpr int MAX_BANKS  = 32;

// DMA_CR Bitmasks (Configurable via Live Spec Mutator)
static constexpr uint32_t DMA_CR_EN          = 0x01; // Bit 0: Enable
static constexpr uint32_t DMA_CR_DIR_M2S     = 0x04; // Bit 2: DRAM -> SRAM
static constexpr uint32_t DMA_CR_BURST16     = 0x10; // Bit 4: 16-Beat Burst
static constexpr uint32_t DMA_CR_RSVD_MBZ    = (1u << HW_RSVD_MBZ_BIT); // Forbidden Reserved Bit!

// DMA_SR Fault & Status Bits
static constexpr uint32_t DMA_SR_DONE             = 0x01;
static constexpr uint32_t DMA_SR_ERR_UNALIGNED    = 0x04; // Bit 2
static constexpr uint32_t DMA_SR_ERR_AXI_DEADLOCK = 0x08; // Bit 3

struct HX9000_State {
    uint32_t DMA_CR        = 0;
    volatile uint32_t DMA_SR = 0;
    uintptr_t DMA_SRC_ADDR = 0;
    uintptr_t DMA_DST_ADDR = 0;
    uintptr_t registered_base_addr = 0;
    uint32_t DMA_LEN       = 0;
    size_t declared_dst_align = static_cast<size_t>(HW_KERNEL_DECLARED_ALIGN);

    bool dmb_called_before_dma  = false;
    bool volatile_status_polled = false;
    bool fault_triggered        = false;
    char fault_code[128]        = "NONE";
    char fault_detail[320]      = "No hardware fault.";

    // Cycle & Performance Counters
    uint64_t total_cycles           = 0;
    uint64_t dma_cycles             = 0;
    uint64_t sram_compute_cycles    = 0;
    uint64_t total_sram_groups      = 0;
    uint64_t conflicted_sram_groups = 0;
    uint64_t bank_access_hist[MAX_BANKS]   = {0};
    uint64_t bank_conflict_hist[MAX_BANKS] = {0};

    // 8x8 Die Thermal Grid (in Celsius) solved via 2D Heat Diffusion PDE
    double die_thermal_grid[8][8];
    double peak_die_temp_c = 42.0;
    double avg_die_temp_c  = 42.0;
    double vdd_droop_mv    = 850.0;
};

inline HX9000_State g_hw;

inline void hw_register_buffer_base(const void* base_ptr) {
    g_hw.registered_base_addr = reinterpret_cast<uintptr_t>(base_ptr);
    g_hw.declared_dst_align = static_cast<size_t>(HW_KERNEL_DECLARED_ALIGN);
}

#define HW_REGISTER_ALIGN(buf) hw_register_buffer_base(&(buf)[0])

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
    return 0x01;
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

    // 1. Check Section 4.2.2: Forbidden Reserved Bit (HW_RSVD_MBZ_BIT)
    if (dma_cr_val & DMA_CR_RSVD_MBZ) {
        g_hw.DMA_SR |= DMA_SR_ERR_AXI_DEADLOCK;
        g_hw.fault_triggered = true;
        g_hw.vdd_droop_mv = 682.4; // Severe voltage droop during AXI bus lock (matches Oscilloscope CSV t=142.4us)
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_AXI_BUS_DEADLOCK_BIT%d", HW_RSVD_MBZ_BIT);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 4.2.2 Violation: DMA_CR (0x%02X) asserts Reserved Bit %d (0x%02X RSVD_MBZ), locking AXI4 WREADY low (CSV t=142.4us, VDD=682mV).",
            dma_cr_val, HW_RSVD_MBZ_BIT, DMA_CR_RSVD_MBZ);
        return false;
    }

    // 2. Check Section 4.2.1: Required Byte Alignment (HW_DMA_ALIGN_BYTES: 64, 128, or 256)
    const uintptr_t align_mask = static_cast<uintptr_t>(HW_DMA_ALIGN_BYTES - 1);
    const uintptr_t dst_offset_bytes = (g_hw.registered_base_addr != 0)
        ? (g_hw.DMA_DST_ADDR - g_hw.registered_base_addr)
        : (g_hw.DMA_DST_ADDR & align_mask);
    const bool ptr_unaligned = ((g_hw.DMA_SRC_ADDR & align_mask) != 0) || ((dst_offset_bytes & align_mask) != 0);
    const bool decl_unaligned = (g_hw.declared_dst_align < static_cast<size_t>(HW_DMA_ALIGN_BYTES));

    if (ptr_unaligned || decl_unaligned) {
        g_hw.DMA_SR |= DMA_SR_ERR_UNALIGNED;
        g_hw.fault_triggered = true;
        g_hw.vdd_droop_mv = 710.0;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_DMA_UNALIGNED_%dB", HW_DMA_ALIGN_BYTES);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 4.2.1 Violation: DMA buffer requires %d-byte alignment (alignas(%d)), but kernel declared alignas(%zu) with byte offset +%llu.",
            HW_DMA_ALIGN_BYTES, HW_DMA_ALIGN_BYTES, g_hw.declared_dst_align, (unsigned long long)dst_offset_bytes);
        return false;
    }

    // 3. Calculate Calibrated DMA Bus Cycles (Section 4.2.2: BURST_16 vs Single-Beat)
    // 256x256 floats = 262,144 bytes = 16,384 beats (128-bit AXI4 bus width)
    uint64_t step_dma_cycles = (dma_cr_val & DMA_CR_BURST16)
        ? ((static_cast<uint64_t>(bytes) * 39701ULL) / 262144ULL)   // 16-beat pipelined burst: 39,701 cycles
        : ((static_cast<uint64_t>(bytes) * 490151ULL) / 262144ULL); // Single-beat bus arbitration: 490,151 cycles
    g_hw.dma_cycles += step_dma_cycles;
    g_hw.total_cycles += step_dma_cycles;

    // 4. Perform transfer (Simulate Section 5.1.1 Stale L1 Cache if hw_dmb() omitted)
    const size_t count = bytes / sizeof(float);
    if (g_hw.dmb_called_before_dma) {
        std::memcpy(dst_ptr, src_ptr, bytes);
        g_hw.vdd_droop_mv = 842.0;
    } else {
        // Stale cache line corruption: 7 out of 8 words per cache line are stale (CSV t=418.0us)
        for (size_t i = 0; i < count; ++i) {
            dst_ptr[i] = (i % 8 == 0) ? src_ptr[i] : -999.0f;
        }
        g_hw.vdd_droop_mv = 815.0;
    }

    g_hw.DMA_SR = DMA_SR_DONE;
    return true;
}

// Simulate N-Lane Vector Load/Store on N-Bank Interleaved Local SRAM (Section 6.4)
inline void hw_record_sram_Gather(const int* elem_indices, int num_lanes) {
    int bank_hits[MAX_BANKS] = {0};
    int max_way_conflict = 1;

    for (int lane = 0; lane < num_lanes; ++lane) {
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
        // Calibrated pipeline stall penalty: 12 cycles per way conflict
        uint64_t penalty = static_cast<uint64_t>(max_way_conflict) * 12ULL;
        g_hw.sram_compute_cycles += penalty;
        g_hw.total_cycles += penalty;
    } else {
        uint64_t norm_scale = (NUM_BANKS == 16) ? 4ULL : 4ULL;
        g_hw.sram_compute_cycles += norm_scale;
        g_hw.total_cycles += norm_scale;
    }
}

// Backward-compatible wrapper for 8-lane or N-lane vector access
inline void hw_record_8lane_sram_access(const int* elem_indices) {
    hw_record_sram_Gather(elem_indices, TILE_DIM);
}

#endif // HW_EMULATOR_HARNESS_HPP
