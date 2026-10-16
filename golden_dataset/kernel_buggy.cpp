#include "hw_emulator_harness.hpp"

// ============================================================================
// Target Kernel: HX-9000 Tensor Transpose & Fused Scale-Bias Kernel
// Computes: dram_out[c, r] = dram_in[r, c] * alpha + 1.0f
// ============================================================================

void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha) {
    // [BUG 1a] Unaligned local SRAM scratchpad buffer (Violates Manual Sec 4.2.1: 64-byte alignment)
    alignas(64) static float raw_backing_store[MATRIX_DIM * MATRIX_DIM + 16];
    float* sram_staging = &raw_backing_store[3]; // Offset by 12 bytes -> (addr & 0x3F != 0)!

    // [BUG 1b] DMA_CR sets Bit 7 (0x80 RSVD_MBZ) -> Triggers ERR_AXI_BUS_DEADLOCK (Sec 4.2.2)
    // Also misses BURST_16 (0x10)
    const uint32_t dma_ctrl_flags = 0x85; // 0x80 (RSVD_MBZ) | 0x04 (DIR_M2S) | 0x01 (EN)

    // [BUG 2a] Missing hw_dmb() Data Memory Barrier before DMA trigger (Violates Sec 5.1.1)
    if (!hw_trigger_dma(dram_in, sram_staging, MATRIX_DIM * MATRIX_DIM * sizeof(float), dma_ctrl_flags)) {
        return; // Aborted due to hardware fault
    }

    // [BUG 2b] Polling DMA_SR without volatile qualification (Violates Sec 5.1.2)
    while ((hw_poll_status_cached() & DMA_SR_DONE) == 0) {
        // Busy wait
    }

    // [BUG 3] 8x8 Local SRAM Tile without +1 row padding (Stride = 8 floats = 32 bytes)
    // Violates Manual Sec 6.4.2: Column-wise access hits the exact same SRAM bank across all 8 rows!
    float local_tile[TILE_DIM][TILE_DIM];
    const int tile_row_stride = TILE_DIM; // 8 floats (32 bytes) -> 8-way Bank Conflict!

    for (int tr = 0; tr < MATRIX_DIM; tr += TILE_DIM) {
        for (int tc = 0; tc < MATRIX_DIM; tc += TILE_DIM) {

            // Step A: Load 8x8 tile row-wise into Local 8-Bank SRAM (1 vector cycle per row)
            for (int r = 0; r < TILE_DIM; ++r) {
                for (int c = 0; c < TILE_DIM; ++c) {
                    local_tile[r][c] = sram_staging[(tr + r) * MATRIX_DIM + (tc + c)];
                }
            }
            int row_init_lanes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
            hw_record_8lane_sram_access(row_init_lanes);

            // Step B: 8-Lane Column-Wise Transpose & Scale-Bias Read from Local 8-Bank SRAM
            // Because tile_row_stride == 8, (r * 8 + c) % 8 == c for all r=0..7 (8-way Bank Conflict!)
            for (int c = 0; c < TILE_DIM; ++c) {
                int col_lanes[8];
                for (int r = 0; r < TILE_DIM; ++r) {
                    col_lanes[r] = r * tile_row_stride + c;
                    float v = local_tile[r][c];
                    dram_out[(tc + c) * MATRIX_DIM + (tr + r)] = v * alpha + 1.0f;
                }
                // Record column-gather micro-ops across the vector pipeline
                hw_record_8lane_sram_access(col_lanes);
                hw_record_8lane_sram_access(col_lanes);
            }
        }
    }
}
