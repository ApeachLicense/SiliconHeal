#ifndef HW_EMULATOR_HARNESS_HPP
#define HW_EMULATOR_HARNESS_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================================
// HX-9000 Hardware Sandbox Emulator (v3.1 - Instrumented SramTile & Anti-Cheat)
// Supports Compile-Time Spec Mutation via Macros (-DHW_*) & static_assert
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
static constexpr int TILE_DIM   = HW_NUM_BANKS;        // Vector tile dimension matches bank width (8 or 16)
static constexpr int NUM_BANKS  = HW_NUM_BANKS;        // 8-Bank or 16-Bank Interleaved Local SRAM
static constexpr int MAX_BANKS  = 32;

// Compile-Time Hardware Constraint Guards
static_assert(HW_DMA_ALIGN_BYTES == 64 || HW_DMA_ALIGN_BYTES == 128 || HW_DMA_ALIGN_BYTES == 256,
              "HW_DMA_ALIGN_BYTES must be 64, 128, or 256 bytes.");
static_assert(HW_NUM_BANKS == 8 || HW_NUM_BANKS == 16,
              "HW_NUM_BANKS must be 8 or 16 interleaved banks.");

// DMA_CR Bitmasks (Configurable via Spec Mutator)
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
    bool sram_bypass_detected   = false;
    char fault_code[128]        = "NONE";
    char fault_detail[360]      = "No hardware fault.";

    // Cycle & Microarchitectural Counters (Observed by Harness, NOT self-reported)
    uint64_t total_cycles             = 0;
    uint64_t dma_cycles               = 0;
    uint64_t sram_compute_cycles      = 0;
    uint64_t total_sram_groups        = 0;
    uint64_t conflicted_sram_groups   = 0;
    uint64_t read_sram_groups         = 0;
    uint64_t conflicted_read_groups   = 0;
    uint64_t strided_transpose_groups = 0;
    uint64_t sram_word_writes         = 0;
    uint64_t sram_word_reads          = 0;
    int observed_tile_rows            = 0;
    int observed_tile_cols            = 0;
    int max_way_conflict_observed     = 1;
    int dirty_l1_lines_bypassed       = 56;
    uint64_t bank_access_hist[MAX_BANKS]   = {0};
    uint64_t bank_conflict_hist[MAX_BANKS] = {0};

    // Wavefront gather accumulator (buffers TILE_DIM consecutive accesses from SramTile)
    int pending_wavefront_words[MAX_BANKS] = {0};
    int pending_wavefront_count            = 0;
    bool pending_wavefront_is_read         = false;

    // Analytical PDN & 8x8 Die Thermal Grid (Gauss-Seidel FD-PDE driven by bank counters)
    double die_thermal_grid[8][8];
    double peak_die_temp_c           = 45.0;
    double avg_die_temp_c            = 45.0;
    double vdd_droop_mv              = 850.0;
    double pdn_di_dt_spike_a_per_ns  = 1.0;
    double bank0_power_density_w_mm2 = 7.94;
};

inline HX9000_State g_hw;

inline void hw_register_buffer_base(const void* base_ptr) {
    g_hw.registered_base_addr = reinterpret_cast<uintptr_t>(base_ptr);
    g_hw.declared_dst_align = static_cast<size_t>(HW_KERNEL_DECLARED_ALIGN);
}

#define HW_REGISTER_ALIGN(buf) hw_register_buffer_base(&(buf)[0])

inline void hw_mark_cache_dirty() {
    g_hw.dmb_called_before_dma = false;
    g_hw.dirty_l1_lines_bypassed = 56;
}

// Section 5.1.1: Mandatory Data Memory Barrier before DMA trigger
inline void hw_dmb() {
    g_hw.dmb_called_before_dma = true;
    g_hw.dirty_l1_lines_bypassed = 0;
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

// Internal crossbar wavefront evaluator (called automatically by SramTile<T, R, C>)
inline void hw_commit_wavefront_group(const int* word_offsets, int num_lanes, bool is_read) {
    int bank_hits[MAX_BANKS] = {0};
    int max_way_conflict = 1;
    bool is_strided_gather = false;

    for (int lane = 0; lane < num_lanes; ++lane) {
        int bank = ((word_offsets[lane]) % NUM_BANKS + NUM_BANKS) % NUM_BANKS;
        bank_hits[bank]++;
        g_hw.bank_access_hist[bank]++;
        if (bank_hits[bank] > max_way_conflict) {
            max_way_conflict = bank_hits[bank];
        }
        if (lane > 0) {
            int step = std::abs(word_offsets[lane] - word_offsets[lane - 1]);
            if (step >= TILE_DIM) {
                is_strided_gather = true;
            }
        }
    }

    if (is_strided_gather) {
        g_hw.strided_transpose_groups++;
    }

    if (max_way_conflict > g_hw.max_way_conflict_observed) {
        g_hw.max_way_conflict_observed = max_way_conflict;
    }

    g_hw.total_sram_groups++;
    if (is_read) {
        g_hw.read_sram_groups++;
    }

    // Base pipelined SRAM + vector FMA execution: 2 cycles per wavefront
    uint64_t wave_cycles = 2ULL;

    if (max_way_conflict > 1) {
        g_hw.conflicted_sram_groups++;
        if (is_read) {
            g_hw.conflicted_read_groups++;
        }
        for (int b = 0; b < NUM_BANKS; ++b) {
            if (bank_hits[b] > 1) {
                g_hw.bank_conflict_hist[b] += static_cast<uint64_t>(bank_hits[b] - 1);
            }
        }
        // Crossbar bank-conflict serialization penalty:
        // Each colliding way beyond 1 stalls the vector pipeline by (168 / TILE_DIM) * ways cycles
        uint64_t stall_per_way = static_cast<uint64_t>((192 + TILE_DIM - 1) / TILE_DIM);
        wave_cycles += static_cast<uint64_t>(max_way_conflict - 1) * stall_per_way;
    }

    g_hw.sram_compute_cycles += wave_cycles;
    g_hw.total_cycles += wave_cycles;
}

inline void hw_flush_pending_wavefront() {
    if (g_hw.pending_wavefront_count > 0) {
        hw_commit_wavefront_group(
            g_hw.pending_wavefront_words,
            g_hw.pending_wavefront_count,
            g_hw.pending_wavefront_is_read
        );
        g_hw.pending_wavefront_count = 0;
    }
}

inline void hw_observe_sram_word_access(int word_offset, bool is_read) {
    if (is_read) {
        g_hw.sram_word_reads++;
    } else {
        g_hw.sram_word_writes++;
    }

    if (g_hw.pending_wavefront_count > 0 && g_hw.pending_wavefront_is_read != is_read) {
        hw_commit_wavefront_group(
            g_hw.pending_wavefront_words,
            g_hw.pending_wavefront_count,
            g_hw.pending_wavefront_is_read
        );
        g_hw.pending_wavefront_count = 0;
    }

    g_hw.pending_wavefront_is_read = is_read;
    g_hw.pending_wavefront_words[g_hw.pending_wavefront_count++] = word_offset;

    if (g_hw.pending_wavefront_count == TILE_DIM) {
        hw_commit_wavefront_group(g_hw.pending_wavefront_words, TILE_DIM, is_read);
        g_hw.pending_wavefront_count = 0;
    }
}

// ============================================================================
// Instrumented Hardware SRAM Tile Wrapper: SramTile<T, Rows, Cols>
// Observes every read and write address directly inside the emulator harness.
// Kernels CANNOT hide bank conflicts by omitting self-reporting calls.
// ============================================================================
template <typename T, int Rows, int Cols>
class SramTile {
    static_assert(Rows >= TILE_DIM && Cols >= TILE_DIM,
                  "SramTile dimensions must be at least TILE_DIM x TILE_DIM.");
private:
    T storage_[Rows * Cols];

public:
    SramTile() {
        g_hw.observed_tile_rows = Rows;
        g_hw.observed_tile_cols = Cols;
        std::memset(storage_, 0, sizeof(storage_));
    }

    struct ElemProxy {
        T* ptr;
        int word_offset;

        // Intercept SRAM Write
        ElemProxy& operator=(T val) {
            hw_observe_sram_word_access(word_offset, false);
            *ptr = val;
            return *this;
        }

        ElemProxy& operator=(const ElemProxy& other) {
            T val = static_cast<T>(other);
            hw_observe_sram_word_access(word_offset, false);
            *ptr = val;
            return *this;
        }

        // Intercept SRAM Read
        operator T() const {
            hw_observe_sram_word_access(word_offset, true);
            return *ptr;
        }
    };

    // 2D access operator: tile(r, c)
    inline ElemProxy operator()(int r, int c) {
        const int word_offset = r * Cols + c;
        return ElemProxy{&storage_[word_offset], word_offset};
    }

    inline T operator()(int r, int c) const {
        const int word_offset = r * Cols + c;
        hw_observe_sram_word_access(word_offset, true);
        return storage_[word_offset];
    }

    // Support 1D bracket indexing `tile[r][c]` so existing C++ array syntax is also instrumented!
    struct RowProxy {
        T* row_ptr;
        int row_idx;

        inline ElemProxy operator[](int c) {
            const int word_offset = row_idx * Cols + c;
            return ElemProxy{&row_ptr[c], word_offset};
        }

        inline T operator[](int c) const {
            const int word_offset = row_idx * Cols + c;
            hw_observe_sram_word_access(word_offset, true);
            return row_ptr[c];
        }
    };

    inline RowProxy operator[](int r) {
        return RowProxy{&storage_[r * Cols], r};
    }

    inline const RowProxy operator[](int r) const {
        return RowProxy{const_cast<T*>(&storage_[r * Cols]), r};
    }

    static constexpr int row_stride_words() { return Cols; }
    static constexpr int row_stride_bytes() { return Cols * static_cast<int>(sizeof(T)); }
};

// Legacy self-reporting API kept as no-op when SramTile is active so old code compiles
// without double-counting, while raw-array bypass kernels are caught by sram_word_reads check.
inline void hw_record_sram_Gather(const int* elem_indices, int num_lanes) {
    (void)elem_indices;
    (void)num_lanes;
}

inline void hw_record_8lane_sram_access(const int* elem_indices) {
    (void)elem_indices;
}

// Trigger DMA engine and validate Section 4.2 & Section 5.1
inline bool hw_trigger_dma(const float* src_ptr, float* dst_ptr, uint32_t bytes, uint32_t dma_cr_val) {
    g_hw.DMA_SRC_ADDR = reinterpret_cast<uintptr_t>(src_ptr);
    g_hw.DMA_DST_ADDR = reinterpret_cast<uintptr_t>(dst_ptr);
    g_hw.DMA_LEN      = bytes;
    g_hw.DMA_CR       = dma_cr_val;

    // 1. Check Section 4.2.2 & Errata #E-401: Forbidden Reserved Bit (HW_RSVD_MBZ_BIT)
    if (dma_cr_val & DMA_CR_RSVD_MBZ) {
        g_hw.DMA_SR |= DMA_SR_ERR_AXI_DEADLOCK;
        g_hw.fault_triggered = true;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_AXI_BUS_DEADLOCK_BIT%d", HW_RSVD_MBZ_BIT);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Sec 4.2.2 & Errata #E-401 Violation: DMA_CR (0x%02X) asserts Reserved Bit %d (0x%02X RSVD_MBZ), locking AXI4 write-response state machine (WREADY=0).",
            dma_cr_val, HW_RSVD_MBZ_BIT, DMA_CR_RSVD_MBZ);
        return false;
    }

    // 2. Check Section 4.2.1 & Errata #E-401: Required Byte Alignment (64, 128, or 256)
    const uintptr_t align_mask = static_cast<uintptr_t>(HW_DMA_ALIGN_BYTES - 1);
    const uintptr_t dst_offset_bytes = (g_hw.registered_base_addr != 0)
        ? (g_hw.DMA_DST_ADDR - g_hw.registered_base_addr)
        : (g_hw.DMA_DST_ADDR & align_mask);
    const bool ptr_unaligned = ((g_hw.DMA_SRC_ADDR & align_mask) != 0) || ((dst_offset_bytes & align_mask) != 0);
    const bool decl_unaligned = (g_hw.declared_dst_align < static_cast<size_t>(HW_DMA_ALIGN_BYTES));

    if (ptr_unaligned || decl_unaligned) {
        g_hw.DMA_SR |= DMA_SR_ERR_UNALIGNED;
        g_hw.fault_triggered = true;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_DMA_UNALIGNED_%dB", HW_DMA_ALIGN_BYTES);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Sec 4.2.1 & Errata #E-401 Violation: DMA buffer requires %d-byte alignment (alignas(%d)), but kernel declared alignas(%zu) with byte offset +%llu.",
            HW_DMA_ALIGN_BYTES, HW_DMA_ALIGN_BYTES, g_hw.declared_dst_align, (unsigned long long)dst_offset_bytes);
        return false;
    }

    // 3. Mechanistic AXI4 Bus Cycle Model (64-bit / 8-byte AXI4 data bus @ 200MHz = 2 core cycles/beat):
    // - Single-beat mode (BURST_16=0): 1 beat (2 core cycles) + 13 core cycles bus turn-around = 15 core cycles/beat
    // - 16-beat burst mode (BURST_16=1): 16 beats (32 core cycles) + 8 core cycles burst setup = 40 core cycles/16-beat burst (2.5 cycles/beat)
    const uint64_t num_beats = (static_cast<uint64_t>(bytes) + 7ULL) / 8ULL;
    uint64_t step_dma_cycles = 0;
    if (dma_cr_val & DMA_CR_BURST16) {
        const uint64_t num_bursts = (num_beats + 15ULL) / 16ULL;
        step_dma_cycles = num_bursts * 40ULL;
    } else {
        step_dma_cycles = num_beats * 15ULL;
    }
    g_hw.dma_cycles += step_dma_cycles;
    g_hw.total_cycles += step_dma_cycles;

    // 4. Perform transfer (Simulate Section 5.1.1 & Errata #E-402 Stale L1 Cache if hw_dmb() omitted)
    const size_t count = bytes / sizeof(float);
    if (g_hw.dmb_called_before_dma) {
        std::memcpy(dst_ptr, src_ptr, bytes);
        g_hw.dirty_l1_lines_bypassed = 0;
    } else {
        // Stale cache line corruption: 7 out of 8 words per cache line are stale
        for (size_t i = 0; i < count; ++i) {
            dst_ptr[i] = (i % 8 == 0) ? src_ptr[i] : -999.0f;
        }
        g_hw.dirty_l1_lines_bypassed = 56;
    }

    g_hw.DMA_SR = DMA_SR_DONE;
    return true;
}

#endif // HW_EMULATOR_HARNESS_HPP
