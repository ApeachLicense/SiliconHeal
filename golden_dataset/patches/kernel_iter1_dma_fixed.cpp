#include "hw_emulator_harness.hpp"

// ============================================================================
// [Iteration 1 Patch] Fixed DMA 64-Byte Alignment & Cleared Reserved Bit 7
// Reference: HX-9000 Manual Section 4.2.1 & Section 4.2.2 (Page 14)
// ============================================================================

void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha) {
    // [FIXED Sec 4.2.1] Use 64-byte aligned base pointer (offset 0 -> addr % 64 == 0)
    alignas(64) static float raw_backing_store[MATRIX_DIM * MATRIX_DIM];
    float* sram_staging = &raw_backing_store[0];

    // [FIXED Sec 4.2.2] Cleared Bit 7 (RSVD_MBZ). Using 0x05 (DIR_M2S | EN)
    const uint32_t dma_ctrl_flags = DMA_CR_DIR_M2S | DMA_CR_EN; // 0x05

    // [STILL BUGGY Sec 5.1.1] Missing hw_dmb() Data Memory Barrier before DMA trigger
    if (!hw_trigger_dma(dram_in, sram_staging, MATRIX_DIM * MATRIX_DIM * sizeof(float), dma_ctrl_flags)) {
        return;
    }

    // [STILL BUGGY Sec 5.1.2] Non-volatile status polling
    while ((hw_poll_status_cached() & DMA_SR_DONE) == 0) {
        // Busy wait
    }

    // [STILL UNOPTIMIZED Sec 6.4.2] Unpadded 8x8 tile -> 32-byte stride bank conflict
    float local_tile[TILE_DIM][TILE_DIM];
    const int tile_row_stride = TILE_DIM; // 8 floats

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
