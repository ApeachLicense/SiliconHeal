#include "hw_emulator_harness.hpp"

// Kernel entry point implemented in kernel.cpp
void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha);

int main() {
    // Allocate 64-byte aligned golden reference buffers for test harness
    alignas(64) static float host_in[MATRIX_DIM * MATRIX_DIM];
    alignas(64) static float host_out[MATRIX_DIM * MATRIX_DIM];
    alignas(64) static float golden_out[MATRIX_DIM * MATRIX_DIM];

    const float alpha = 2.5f;

    // Initialize input tensor and mark L1 cache dirty (CPU write)
    hw_mark_cache_dirty();
    for (int r = 0; r < MATRIX_DIM; ++r) {
        for (int c = 0; c < MATRIX_DIM; ++c) {
            float val = static_cast<float>((r * 31 + c * 17) % 97) * 0.25f;
            host_in[r * MATRIX_DIM + c] = val;
            host_out[r * MATRIX_DIM + c] = 0.0f;
        }
    }

    // Compute Golden Reference: Transposed & Scaled Tensor (out[c, r] = in[r, c] * alpha + bias)
    for (int r = 0; r < MATRIX_DIM; ++r) {
        for (int c = 0; c < MATRIX_DIM; ++c) {
            golden_out[c * MATRIX_DIM + r] = host_in[r * MATRIX_DIM + c] * alpha + 1.0f;
        }
    }

    // Execute kernel under test
    run_hx9000_tensor_kernel(host_in, host_out, alpha);

    // Verify correctness against Golden Reference
    int total_elements = MATRIX_DIM * MATRIX_DIM;
    int matched_elements = 0;
    uint32_t crc_golden = 0x811C9DC5u;
    uint32_t crc_actual  = 0x811C9DC5u;

    for (int i = 0; i < total_elements; ++i) {
        uint32_t g_bits = 0, a_bits = 0;
        std::memcpy(&g_bits, &golden_out[i], sizeof(float));
        std::memcpy(&a_bits, &host_out[i], sizeof(float));
        crc_golden = (crc_golden ^ g_bits) * 0x01000193u;
        crc_actual = (crc_actual ^ a_bits) * 0x01000193u;

        if (std::fabs(host_out[i] - golden_out[i]) < 1e-4f) {
            matched_elements++;
        }
    }

    // If non-volatile status polling was used, simulate race condition corruption on tail tiles
    if (!g_hw.fault_triggered && !g_hw.volatile_status_polled) {
        matched_elements = matched_elements / 8;
        crc_actual ^= 0xDEAD0001u;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 5.1 Violation: Missing hw_dmb() barrier and/or non-volatile DMA_SR polling caused stale L1 cache lines.");
    } else if (!g_hw.fault_triggered && matched_elements < total_elements) {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 5.1.1 Violation: Missing hw_dmb() before DMA trigger resulted in stale DRAM read.");
    }

    bool crc_passed = (!g_hw.fault_triggered) && (matched_elements == total_elements) && g_hw.volatile_status_polled;
    double accuracy_pct = g_hw.fault_triggered ? 0.0 : (100.0 * matched_elements / static_cast<double>(total_elements));

    // Calculate Telemetry Metrics (400 MHz Core Clock -> 1 cycle = 2.5 ns, scaled for 256-batch inference frame)
    double bank_conflict_rate = (g_hw.total_sram_groups > 0)
        ? (100.0 * static_cast<double>(g_hw.conflicted_sram_groups) / static_cast<double>(g_hw.total_sram_groups))
        : 0.0;

    // Scale single tensor pass to a 64-layer Edge AI inference frame
    uint64_t frame_cycles = g_hw.total_cycles * 12ULL;
    double latency_ms = g_hw.fault_triggered ? 0.0 : (static_cast<double>(frame_cycles) / 400000.0);
    double throughput_gops = (latency_ms > 0.0) ? (67.108864 / latency_ms) : 0.0;

    const char* overall_status = "ALL_SLA_PASSED";
    if (g_hw.fault_triggered) {
        overall_status = "HARDWARE_FAULT";
    } else if (!crc_passed) {
        overall_status = "DATA_CORRUPTION";
    } else if (latency_ms > 3.5 || bank_conflict_rate > 5.0) {
        overall_status = "SLA_LATENCY_VIOLATION";
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "WARN_SRAM_8BANK_CONFLICT");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 6.4.2 Bottleneck: 32-byte column stride causes %.1f%% 8-way SRAM Bank 0 conflict & single-beat DMA overhead (%.2f ms > 3.50 ms SLA).",
            bank_conflict_rate, latency_ms);
    } else {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "OK_SPEC_COMPLIANT");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "All HX-9000 Rev 4.2 constraints verified: 64B aligned, DMB+Volatile active, BURST_16 enabled, +1 SRAM padding (0%% bank conflict).");
    }

    // Emit JSON Telemetry
    std::printf("{\n");
    std::printf("  \"status\": \"%s\",\n", overall_status);
    std::printf("  \"fault_code\": \"%s\",\n", g_hw.fault_code);
    std::printf("  \"fault_detail\": \"%s\",\n", g_hw.fault_detail);
    std::printf("  \"dma_cr_hex\": \"0x%02X\",\n", g_hw.DMA_CR);
    std::printf("  \"src_aligned_64b\": %s,\n", ((g_hw.DMA_SRC_ADDR & 0x3FULL) == 0) ? "true" : "false");
    std::printf("  \"dst_aligned_64b\": %s,\n", ((g_hw.DMA_DST_ADDR & 0x3FULL) == 0) ? "true" : "false");
    std::printf("  \"dmb_barrier_called\": %s,\n", g_hw.dmb_called_before_dma ? "true" : "false");
    std::printf("  \"volatile_polled\": %s,\n", g_hw.volatile_status_polled ? "true" : "false");
    std::printf("  \"burst16_enabled\": %s,\n", ((g_hw.DMA_CR & DMA_CR_BURST16) != 0) ? "true" : "false");
    std::printf("  \"crc_passed\": %s,\n", crc_passed ? "true" : "false");
    std::printf("  \"crc_expected\": \"0x%08X\",\n", crc_golden);
    std::printf("  \"crc_actual\": \"0x%08X\",\n", crc_actual);
    std::printf("  \"accuracy_pct\": %.2f,\n", accuracy_pct);
    std::printf("  \"frame_cycles\": %llu,\n", (unsigned long long)frame_cycles);
    std::printf("  \"latency_ms\": %.2f,\n", latency_ms);
    std::printf("  \"sla_target_ms\": 3.50,\n");
    std::printf("  \"throughput_gops\": %.2f,\n", throughput_gops);
    std::printf("  \"bank_conflict_rate_pct\": %.2f,\n", bank_conflict_rate);
    std::printf("  \"bank_access_hist\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu],\n",
        (unsigned long long)g_hw.bank_access_hist[0], (unsigned long long)g_hw.bank_access_hist[1],
        (unsigned long long)g_hw.bank_access_hist[2], (unsigned long long)g_hw.bank_access_hist[3],
        (unsigned long long)g_hw.bank_access_hist[4], (unsigned long long)g_hw.bank_access_hist[5],
        (unsigned long long)g_hw.bank_access_hist[6], (unsigned long long)g_hw.bank_access_hist[7]);
    std::printf("  \"bank_conflict_hist\": [%llu, %llu, %llu, %llu, %llu, %llu, %llu, %llu]\n",
        (unsigned long long)g_hw.bank_conflict_hist[0], (unsigned long long)g_hw.bank_conflict_hist[1],
        (unsigned long long)g_hw.bank_conflict_hist[2], (unsigned long long)g_hw.bank_conflict_hist[3],
        (unsigned long long)g_hw.bank_conflict_hist[4], (unsigned long long)g_hw.bank_conflict_hist[5],
        (unsigned long long)g_hw.bank_conflict_hist[6], (unsigned long long)g_hw.bank_conflict_hist[7]);
    std::printf("}\n");

    return (std::strcmp(overall_status, "ALL_SLA_PASSED") == 0) ? 0 : 1;
}
