#include "hw_emulator_harness.hpp"

// ============================================================================
// [Iteration 2 Patch] Added Data Memory Barrier (DMB) & Volatile Status Poll
// Reference: HX-9000 Manual Section 5.1.1 & Section 5.1.2 (Page 27)
// ============================================================================

void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha) {
    // [FIXED Sec 4.2.1] 64-byte aligned base pointer
    alignas(64) static float raw_backing_store[MATRIX_DIM * MATRIX_DIM];
    float* sram_staging = &raw_backing_store[0];

    // [FIXED Sec 4.2.2] Cleared Bit 7 (RSVD_MBZ)
    const uint32_t dma_ctrl_flags = DMA_CR_DIR_M2S | DMA_CR_EN; // 0x05

    // [FIXED Sec 5.1.1] Mandatory Data Memory Barrier to flush dirty L1 cache lines before DMA
    hw_dmb();
    if (!hw_trigger_dma(dram_in, sram_staging, MATRIX_DIM * MATRIX_DIM * sizeof(float), dma_ctrl_flags)) {
        return;
    }

    // [FIXED Sec 5.1.2] Volatile MMIO status register polling
    while ((hw_poll_status_volatile() & DMA_SR_DONE) == 0) {
        // Busy wait until hardware asserts DMA_SR.DONE
    }

    // [STILL UNOPTIMIZED Sec 6.4.2] Unpadded 8x8 tile -> 32-byte stride causes 94.1% Bank Conflict
    float local_tile[TILE_DIM][TILE_DIM];
    const int tile_row_stride = TILE_DIM; // 8 floats (32 bytes)

    for (int tr = 0; tr < MATRIX_DIM; tr += TILE_DIM) {
        for (int tc = 0; tc < MATRIX_DIM; tc += TILE_DIM) {
            for (int r = 0; r < TILE_DIM; ++r) {
                for (int c = 0; c < TILE_DIM; ++c) {
                    local_tile[r][c] = sram_staging[(tr + r) * MATRIX_DIM + (tc + c)];
                }
            }
            int row_init_lanes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            hw_record_8lane_sram_access(row_init_lanes);

            for (int c = 0; c < TILE_DIM; ++c) {
                int col_lanes[8];
                for (int r = 0; r < TILE_DIM; ++r) {
                    col_lanes[r] = r * tile_row_stride + c;
                    float v = local_tile[r][c];
                    dram_out[(tc + c) * MATRIX_DIM + (tr + r)] = v * alpha + 1.0f;
                }
                hw_record_8lane_sram_access(col_lanes);
                hw_record_8lane_sram_access(col_lanes);
            }
        }
    }
}
