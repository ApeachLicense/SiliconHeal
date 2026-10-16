#include "hw_emulator_harness.hpp"

// Kernel entry point implemented in kernel.cpp
void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha);

// Solve 2D Steady-State Heat Diffusion PDE on 8x8 Silicon Die Grid (Gauss-Seidel)
static void solve_die_thermal_pde(bool is_bus_fault, double bank_conflict_pct, bool burst16_on) {
    const double t_ambient = 45.0;
    double power_source[8][8];

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            // Base leakage & logic power across die
            power_source[r][c] = burst16_on ? 1.8 : 2.9;
            g_hw.die_thermal_grid[r][c] = t_ambient + 10.0;
        }
    }

    if (is_bus_fault) {
        // Intense local current crowds at AXI4 Arbiter & DMA Controller macro (top-left die quadrant -> ~95.2C)
        for (int r = 1; r < 5; ++r) {
            for (int c = 1; c < 5; ++c) {
                power_source[r][c] = 28.5;
            }
        }
        power_source[2][2] = 46.8;
        power_source[2][3] = 44.0;
    } else if (bank_conflict_pct > 20.0) {
        // 8-Way / 16-Way SRAM Bank 0 Sense-Amplifier & Crossbar Hotspot (center die macro -> 98.4C)
        double conflict_factor = bank_conflict_pct / 94.12;
        for (int r = 1; r < 7; ++r) {
            power_source[r][2] = 23.8 * conflict_factor;
            power_source[r][3] = 37.1 * conflict_factor;
            power_source[r][4] = 25.6 * conflict_factor;
        }
        power_source[3][3] = 45.9 * conflict_factor;
        power_source[4][3] = 45.0 * conflict_factor;
    } else {
        // Uniformly distributed N-Bank parallel access: low, balanced power density (~63.8C peak)
        for (int r = 1; r < 7; ++r) {
            for (int c = 1; c < 7; ++c) {
                power_source[r][c] = 7.94;
            }
        }
    }

    // 80 iterations of 2D Finite-Difference Gauss-Seidel relaxation:
    // T[r][c] = (T_neighbors + alpha_q * P[r][c] + h_sink * T_amb) / (4 + h_sink)
    const double h_sink = 0.35;
    for (int iter = 0; iter < 80; ++iter) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                double t_up    = (r > 0) ? g_hw.die_thermal_grid[r - 1][c] : (t_ambient + 6.0);
                double t_down  = (r < 7) ? g_hw.die_thermal_grid[r + 1][c] : (t_ambient + 6.0);
                double t_left  = (c > 0) ? g_hw.die_thermal_grid[r][c - 1] : (t_ambient + 6.0);
                double t_right = (c < 7) ? g_hw.die_thermal_grid[r][c + 1] : (t_ambient + 6.0);

                g_hw.die_thermal_grid[r][c] =
                    (t_up + t_down + t_left + t_right + power_source[r][c] + h_sink * t_ambient) / (4.0 + h_sink);
            }
        }
    }

    double peak_t = 0.0;
    double sum_t  = 0.0;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            if (g_hw.die_thermal_grid[r][c] > peak_t) {
                peak_t = g_hw.die_thermal_grid[r][c];
            }
            sum_t += g_hw.die_thermal_grid[r][c];
        }
    }
    g_hw.peak_die_temp_c = peak_t;
    g_hw.avg_die_temp_c  = sum_t / 64.0;
}

int main() {
    static float raw_host_in[MATRIX_DIM * MATRIX_DIM + 128];
    static float raw_host_out[MATRIX_DIM * MATRIX_DIM + 128];
    static float raw_golden_out[MATRIX_DIM * MATRIX_DIM + 128];

    auto align256 = [](float* p) -> float* {
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        uintptr_t aligned = (addr + 255ULL) & ~static_cast<uintptr_t>(255ULL);
        return reinterpret_cast<float*>(aligned);
    };
    float* host_in    = align256(raw_host_in);
    float* host_out   = align256(raw_host_out);
    float* golden_out = align256(raw_golden_out);

    const float alpha = 2.5f;

    hw_mark_cache_dirty();
    for (int r = 0; r < MATRIX_DIM; ++r) {
        for (int c = 0; c < MATRIX_DIM; ++c) {
            float val = static_cast<float>((r * 31 + c * 17) % 97) * 0.25f;
            host_in[r * MATRIX_DIM + c] = val;
            host_out[r * MATRIX_DIM + c] = 0.0f;
        }
    }

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
    uint32_t crc_actual = 0x811C9DC5u;

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

    if (!g_hw.fault_triggered && !g_hw.volatile_status_polled) {
        matched_elements = matched_elements / 8;
        crc_actual ^= 0xDEAD0001u;
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 5.1 Violation: Missing hw_dmb() barrier and/or non-volatile DMA_SR polling caused stale L1 cache lines (CSV t=418.0us: 56 dirty lines).");
    } else if (!g_hw.fault_triggered && matched_elements < total_elements) {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 5.1.1 Violation: Missing hw_dmb() before DMA trigger resulted in stale DRAM read (CSV t=418.0us).");
    }

    bool crc_passed = (!g_hw.fault_triggered) && (matched_elements == total_elements) && g_hw.volatile_status_polled;
    double accuracy_pct = g_hw.fault_triggered ? 0.0 : (100.0 * matched_elements / static_cast<double>(total_elements));

    double bank_conflict_rate = (g_hw.total_sram_groups > 0)
        ? (100.0 * static_cast<double>(g_hw.conflicted_sram_groups) / static_cast<double>(g_hw.total_sram_groups))
        : 0.0;

    // Calibrated 9-Layer Inference Block Cycle Model @ 400 MHz Core Clock
    // Directly proportional to measured g_hw.dma_cycles + g_hw.sram_compute_cycles
    uint64_t weighted_layer_cycles = g_hw.dma_cycles + g_hw.sram_compute_cycles;
    uint64_t frame_cycles = g_hw.fault_triggered ? 0ULL : (weighted_layer_cycles * 9ULL);
    double latency_ms = g_hw.fault_triggered ? 0.0 : (static_cast<double>(frame_cycles) / 400000.0);
    double throughput_gops = (latency_ms > 0.0) ? (67.108864 / latency_ms) : 0.0;

    bool burst16_on = ((g_hw.DMA_CR & DMA_CR_BURST16) != 0);
    solve_die_thermal_pde(g_hw.fault_triggered, bank_conflict_rate, burst16_on);

    const char* overall_status = "ALL_SLA_PASSED";
    if (g_hw.fault_triggered) {
        overall_status = "HARDWARE_FAULT";
    } else if (!crc_passed) {
        overall_status = "DATA_CORRUPTION";
    } else if (latency_ms > HW_SLA_TARGET_MS || bank_conflict_rate > 5.0 || !burst16_on) {
        overall_status = "SLA_LATENCY_VIOLATION";
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "WARN_SRAM_%dBANK_CONFLICT", NUM_BANKS);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Section 6.4.2 Bottleneck: Unpadded %d-float stride causes %.2f%% %d-Bank conflict & %.1fC die hotspot (%.2f ms > %.2f ms SLA; CSV t=1120.5us).",
            TILE_DIM, bank_conflict_rate, NUM_BANKS, g_hw.peak_die_temp_c, latency_ms, static_cast<double>(HW_SLA_TARGET_MS));
    } else {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "OK_SPEC_COMPLIANT");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "All HX-9000 specs verified (Align=%dB, RSVD_Bit=%d clear, DMB+Volatile active, BURST_16=1, %d-Bank +1 Pad -> 0%% conflict, PeakTemp=%.1fC).",
            HW_DMA_ALIGN_BYTES, HW_RSVD_MBZ_BIT, NUM_BANKS, g_hw.peak_die_temp_c);
    }

    const uintptr_t align_mask = static_cast<uintptr_t>(HW_DMA_ALIGN_BYTES - 1);
    bool dst_aligned_ok = ((g_hw.DMA_DST_ADDR & align_mask) == 0) &&
                          (g_hw.declared_dst_align >= static_cast<size_t>(HW_DMA_ALIGN_BYTES));

    // Emit JSON Telemetry including Spec Config & 8x8 Die Thermal Grid
    std::printf("{\n");
    std::printf("  \"status\": \"%s\",\n", overall_status);
    std::printf("  \"fault_code\": \"%s\",\n", g_hw.fault_code);
    std::printf("  \"fault_detail\": \"%s\",\n", g_hw.fault_detail);
    std::printf("  \"spec_config\": {\"dma_align_bytes\": %d, \"rsvd_mbz_bit\": %d, \"num_banks\": %d, \"sla_target_ms\": %.2f},\n",
        HW_DMA_ALIGN_BYTES, HW_RSVD_MBZ_BIT, NUM_BANKS, static_cast<double>(HW_SLA_TARGET_MS));
    std::printf("  \"dma_cr_hex\": \"0x%02X\",\n", g_hw.DMA_CR);
    std::printf("  \"declared_dst_align\": %zu,\n", g_hw.declared_dst_align);
    std::printf("  \"src_aligned_ok\": %s,\n", ((g_hw.DMA_SRC_ADDR & align_mask) == 0) ? "true" : "false");
    std::printf("  \"dst_aligned_ok\": %s,\n", dst_aligned_ok ? "true" : "false");
    std::printf("  \"src_aligned_64b\": %s,\n", ((g_hw.DMA_SRC_ADDR & align_mask) == 0) ? "true" : "false");
    std::printf("  \"dst_aligned_64b\": %s,\n", dst_aligned_ok ? "true" : "false");
    std::printf("  \"dmb_barrier_called\": %s,\n", g_hw.dmb_called_before_dma ? "true" : "false");
    std::printf("  \"volatile_polled\": %s,\n", g_hw.volatile_status_polled ? "true" : "false");
    std::printf("  \"burst16_enabled\": %s,\n", burst16_on ? "true" : "false");
    std::printf("  \"crc_passed\": %s,\n", crc_passed ? "true" : "false");
    std::printf("  \"crc_expected\": \"0x%08X\",\n", crc_golden);
    std::printf("  \"crc_actual\": \"0x%08X\",\n", crc_actual);
    std::printf("  \"accuracy_pct\": %.2f,\n", accuracy_pct);
    std::printf("  \"frame_cycles\": %llu,\n", (unsigned long long)frame_cycles);
    std::printf("  \"latency_ms\": %.2f,\n", latency_ms);
    std::printf("  \"sla_target_ms\": %.2f,\n", static_cast<double>(HW_SLA_TARGET_MS));
    std::printf("  \"throughput_gops\": %.2f,\n", throughput_gops);
    std::printf("  \"bank_conflict_rate_pct\": %.2f,\n", bank_conflict_rate);
    std::printf("  \"vdd_droop_mv\": %.1f,\n", g_hw.vdd_droop_mv);
    std::printf("  \"peak_die_temp_c\": %.1f,\n", g_hw.peak_die_temp_c);
    std::printf("  \"avg_die_temp_c\": %.1f,\n", g_hw.avg_die_temp_c);

    std::printf("  \"bank_access_hist\": [");
    for (int b = 0; b < NUM_BANKS; ++b) {
        std::printf("%llu%s", (unsigned long long)g_hw.bank_access_hist[b], (b + 1 < NUM_BANKS) ? ", " : "");
    }
    std::printf("],\n");

    std::printf("  \"bank_conflict_hist\": [");
    for (int b = 0; b < NUM_BANKS; ++b) {
        std::printf("%llu%s", (unsigned long long)g_hw.bank_conflict_hist[b], (b + 1 < NUM_BANKS) ? ", " : "");
    }
    std::printf("],\n");

    std::printf("  \"die_thermal_grid_c\": [\n");
    for (int r = 0; r < 8; ++r) {
        std::printf("    [");
        for (int c = 0; c < 8; ++c) {
            std::printf("%.1f%s", g_hw.die_thermal_grid[r][c], (c < 7) ? ", " : "");
        }
        std::printf("]%s\n", (r < 7) ? "," : "");
    }
    std::printf("  ]\n");
    std::printf("}\n");

    return (std::strcmp(overall_status, "ALL_SLA_PASSED") == 0) ? 0 : 1;
}
