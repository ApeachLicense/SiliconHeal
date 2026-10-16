#include "hw_emulator_harness.hpp"

// ============================================================================
// Target Kernel: HX-9000 Tensor Transpose & Fused Scale-Bias Kernel
// Computes: dram_out[c, r] = dram_in[r, c] * alpha + 1.0f
// ============================================================================

void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha) {
    // [BUG 1a] Inadequate alignment & unaligned offset pointer (Violates Manual Sec 4.2.1)
    alignas(64) static float raw_backing_store[MATRIX_DIM * MATRIX_DIM + 16];
    HW_REGISTER_ALIGN(raw_backing_store);
    float* sram_staging = &raw_backing_store[3]; // Offset by +12 bytes -> Unaligned!

    // [BUG 1b] DMA_CR sets Reserved Bit (0xC5 sets both Bit 7 and Bit 6) -> Triggers ERR_AXI_BUS_DEADLOCK (Sec 4.2.2)
    // Also misses BURST_16 (0x10)
    const uint32_t dma_ctrl_flags = 0xC5;

    // [BUG 2a] Missing hw_dmb() Data Memory Barrier before DMA trigger (Violates Sec 5.1.1)
    if (!hw_trigger_dma(dram_in, sram_staging, MATRIX_DIM * MATRIX_DIM * sizeof(float), dma_ctrl_flags)) {
        return; // Aborted due to hardware fault
    }

    // [BUG 2b] Polling DMA_SR without volatile qualification (Violates Sec 5.1.2)
    while ((hw_poll_status_cached() & DMA_SR_DONE) == 0) {
        // Busy wait
    }

    // [BUG 3] Unpadded Local SRAM Tile: SramTile<float, TILE_DIM, TILE_DIM> (Row Stride = TILE_DIM words)
    // Violates Manual Sec 6.4.2: Column-wise read hits the exact same SRAM bank across all rows!
    SramTile<float, TILE_DIM, TILE_DIM> local_tile;

    for (int tr = 0; tr < MATRIX_DIM; tr += TILE_DIM) {
        for (int tc = 0; tc < MATRIX_DIM; tc += TILE_DIM) {

            // Step A: Load tile row-wise into Local Interleaved SRAM
            for (int r = 0; r < TILE_DIM; ++r) {
                for (int c = 0; c < TILE_DIM; ++c) {
                    local_tile(r, c) = sram_staging[(tr + r) * MATRIX_DIM + (tc + c)];
                }
            }

            // Step B: Column-Wise Transpose & Scale-Bias Read from Local SRAM
            for (int c = 0; c < TILE_DIM; ++c) {
                for (int r = 0; r < TILE_DIM; ++r) {
                    float v = local_tile(r, c);
                    dram_out[(tc + c) * MATRIX_DIM + (tr + r)] = v * alpha + 1.0f;
                }
            }
        }
    }
}
