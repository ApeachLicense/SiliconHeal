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
        // Intense local current crowds at AXI4 Arbiter & DMA Controller macro (top-left die quadrant -> ~94.2C)
        g_hw.bank0_power_density_w_mm2 = 46.8;
        for (int r = 1; r < 5; ++r) {
            for (int c = 1; c < 5; ++c) {
                power_source[r][c] = 28.5;
            }
        }
        power_source[2][2] = 46.8;
        power_source[2][3] = 44.0;
    } else if (bank_conflict_pct > 20.0) {
        // N-Way SRAM Bank 0 Sense-Amplifier & Crossbar Wavefront Serialization Hotspot (97.7C ~ 99.2C)
        double conflict_factor = bank_conflict_pct / 94.12;
        g_hw.bank0_power_density_w_mm2 = 45.9 * conflict_factor;
        for (int r = 1; r < 7; ++r) {
            power_source[r][2] = 23.8 * conflict_factor;
            power_source[r][3] = 37.1 * conflict_factor;
            power_source[r][4] = 25.6 * conflict_factor;
        }
        power_source[3][3] = 45.9 * conflict_factor;
        power_source[4][3] = 45.0 * conflict_factor;
    } else {
        // Uniformly distributed N-Bank parallel access: low, balanced power density (~62.2C peak)
        g_hw.bank0_power_density_w_mm2 = 7.94;
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
            "Sec 5.1 & Errata #E-402 Violation: Missing hw_dmb() barrier and non-volatile DMA_SR polling left 56 dirty L1 cache lines unflushed (CSV t=418.00us).");
    } else if (!g_hw.fault_triggered && matched_elements < total_elements) {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Sec 5.1.1 & Errata #E-402 Violation: Missing hw_dmb() before DMA trigger resulted in stale DRAM read (CSV t=418.00us).");
    }

    bool crc_passed = (!g_hw.fault_triggered) && (matched_elements == total_elements) && g_hw.volatile_status_polled;
    double accuracy_pct = g_hw.fault_triggered ? 0.0 : (100.0 * matched_elements / static_cast<double>(total_elements));

    double bank_conflict_rate = (g_hw.total_sram_groups > 0)
        ? (100.0 * static_cast<double>(g_hw.conflicted_sram_groups) / static_cast<double>(g_hw.total_sram_groups))
        : 0.0;

    // Calibrated 9-Layer Inference Block Cycle Model @ 400 MHz Core Clock
    uint64_t weighted_layer_cycles = g_hw.dma_cycles + g_hw.sram_compute_cycles;
    uint64_t frame_cycles = g_hw.fault_triggered ? 0ULL : (weighted_layer_cycles * 9ULL);
    double latency_ms = g_hw.fault_triggered ? 0.0 : (static_cast<double>(frame_cycles) / 400000.0);
    double throughput_gops = (latency_ms > 0.0) ? (67.108864 / latency_ms) : 0.0;

    bool burst16_on = ((g_hw.DMA_CR & DMA_CR_BURST16) != 0);
    solve_die_thermal_pde(g_hw.fault_triggered, bank_conflict_rate, burst16_on);

    // Real-Time WCET (Worst-Case Execution Time) & Jitter Modeling (FreeRTOS/QEMU Firmware Validation Standard)
    double wcet_ms = 0.0;
    double jitter_us = 0.0;
    if (g_hw.fault_triggered) {
        wcet_ms = 0.0;       // Halted on bus deadlock
        jitter_us = 9999.0;  // Unbounded lock jitter
    } else if (!crc_passed) {
        wcet_ms = latency_ms * 1.18; // Non-deterministic L1 eviction + single-beat arbitration + bank stalls
        jitter_us = 412.5;
    } else if (bank_conflict_rate > 5.0 || !burst16_on) {
        wcet_ms = latency_ms * 1.14; // Single-beat bus arbitration + N-way wavefront serialization variance
        jitter_us = 284.6;
    } else {
        wcet_ms = latency_ms * 1.03; // Deterministic 16-beat pipelined burst + 0% conflict coprime access
        jitter_us = 4.2;             // Strictly within <= 25.0 us hard real-time jitter ceiling
    }

    const char* overall_status = "ALL_SLA_PASSED";
    if (g_hw.fault_triggered) {
        overall_status = "HARDWARE_FAULT";
    } else if (!crc_passed) {
        overall_status = "DATA_CORRUPTION";
    } else if (latency_ms > HW_SLA_TARGET_MS || wcet_ms > HW_SLA_TARGET_MS || bank_conflict_rate > 5.0 || !burst16_on) {
        overall_status = "SLA_LATENCY_VIOLATION";
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "WARN_SRAM_%dBANK_CONFLICT", NUM_BANKS);
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Sec 6.4.2 & Errata #E-403 Bottleneck: Unpadded %d-float stride causes %.2f%% %d-Bank wavefront serialization & %.1fC Bank 0 hotspot (Latency %.2fms, WCET %.2fms > %.2fms SLA).",
            TILE_DIM, bank_conflict_rate, NUM_BANKS, g_hw.peak_die_temp_c, latency_ms, wcet_ms, static_cast<double>(HW_SLA_TARGET_MS));
    } else {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "OK_SPEC_COMPLIANT");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "All HX-9000 specs & Errata #E-401..#E-403 verified (Align=%dB, Bit %d clear, DMB+Volatile, BURST_16=1, %d-Bank +1 Pad -> 0%% conflict, WCET=%.2fms, Jitter=%.1fus, PeakTemp=%.1fC).",
            HW_DMA_ALIGN_BYTES, HW_RSVD_MBZ_BIT, NUM_BANKS, wcet_ms, jitter_us, g_hw.peak_die_temp_c);
    }

    const uintptr_t align_mask = static_cast<uintptr_t>(HW_DMA_ALIGN_BYTES - 1);
    const uintptr_t dst_offset_bytes = (g_hw.registered_base_addr != 0)
        ? (g_hw.DMA_DST_ADDR - g_hw.registered_base_addr)
        : (g_hw.DMA_DST_ADDR & align_mask);
    bool dst_aligned_ok = ((dst_offset_bytes & align_mask) == 0) &&
                          (g_hw.declared_dst_align >= static_cast<size_t>(HW_DMA_ALIGN_BYTES));

    int wavefront_ways = g_hw.fault_triggered ? 0 : g_hw.max_way_conflict_observed;

    // Emit JSON Telemetry including Spec Config, Real-Time WCET/Jitter, Causal Graph & 8x8 Die Thermal Grid
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
    std::printf("  \"dirty_l1_lines_bypassed\": %d,\n", g_hw.dirty_l1_lines_bypassed);
    std::printf("  \"volatile_polled\": %s,\n", g_hw.volatile_status_polled ? "true" : "false");
    std::printf("  \"burst16_enabled\": %s,\n", burst16_on ? "true" : "false");
    std::printf("  \"crc_passed\": %s,\n", crc_passed ? "true" : "false");
    std::printf("  \"crc_expected\": \"0x%08X\",\n", crc_golden);
    std::printf("  \"crc_actual\": \"0x%08X\",\n", crc_actual);
    std::printf("  \"accuracy_pct\": %.2f,\n", accuracy_pct);
    std::printf("  \"frame_cycles\": %llu,\n", (unsigned long long)frame_cycles);
    std::printf("  \"latency_ms\": %.2f,\n", latency_ms);
    std::printf("  \"wcet_ms\": %.2f,\n", wcet_ms);
    std::printf("  \"jitter_us\": %.1f,\n", jitter_us);
    std::printf("  \"sla_target_ms\": %.2f,\n", static_cast<double>(HW_SLA_TARGET_MS));
    std::printf("  \"throughput_gops\": %.2f,\n", throughput_gops);
    std::printf("  \"bank_conflict_rate_pct\": %.2f,\n", bank_conflict_rate);
    std::printf("  \"wavefront_serialization_ways\": %d,\n", wavefront_ways);
    std::printf("  \"vdd_droop_mv\": %.1f,\n", g_hw.vdd_droop_mv);
    std::printf("  \"pdn_di_dt_spike_a_per_ns\": %.1f,\n", g_hw.pdn_di_dt_spike_a_per_ns);
    std::printf("  \"bank0_power_density_w_mm2\": %.2f,\n", g_hw.bank0_power_density_w_mm2);
    std::printf("  \"peak_die_temp_c\": %.1f,\n", g_hw.peak_die_temp_c);
    std::printf("  \"avg_die_temp_c\": %.1f,\n", g_hw.avg_die_temp_c);

    // Physical Causal Graph Separation (PDN Transient Voltage Droop vs. Wavefront Serialization Thermal Hotspot)
    std::printf("  \"causal_graph\": {\n");
    std::printf("    \"voltage_droop_cause\": {\n");
    std::printf("      \"phenomenon\": \"VDD_CORE = %.1f mV (Nominal 850.0 mV)\",\n", g_hw.vdd_droop_mv);
    std::printf("      \"physical_mechanism\": \"PDN Transient Inductive Drop (V_droop = L_pkg * di/dt + I * R_pdn)\",\n");
    std::printf("      \"di_dt_spike_a_per_ns\": %.1f,\n", g_hw.pdn_di_dt_spike_a_per_ns);
    std::printf("      \"root_trigger\": \"%s\"\n",
        g_hw.fault_triggered
            ? "Unaligned DMA address trap & DMA_CR[RSVD_MBZ] bus deadlock causing 200MHz AXI4 arbiter retry current spikes (Errata #E-401)"
            : "Clean aligned DMA transfer with zero arbiter deadlock oscillation");
    std::printf("    },\n");
    std::printf("    \"thermal_hotspot_cause\": {\n");
    std::printf("      \"phenomenon\": \"Peak Die Temp = %.1f C (Avg %.1f C)\",\n", g_hw.peak_die_temp_c, g_hw.avg_die_temp_c);
    std::printf("      \"physical_mechanism\": \"2D Steady-State Heat Conduction (div(k_Si * grad(T)) = -P_density)\",\n");
    std::printf("      \"bank0_power_density_w_mm2\": %.2f,\n", g_hw.bank0_power_density_w_mm2);
    std::printf("      \"wavefront_serialization_ways\": %d,\n", wavefront_ways);
    std::printf("      \"root_trigger\": \"%s\"\n",
        (bank_conflict_rate > 20.0)
            ? "Unpadded power-of-two row stride routes all column-gather vector lanes to Bank 0, causing N-way wavefront serialization stalls and localized sense-amp heating (Errata #E-403)"
            : (g_hw.fault_triggered
                ? "AXI4 arbiter bus lock localized current crowding in DMA controller macro"
                : "Coprime +1 padded row stride (gcd(TILE_DIM+1, NUM_BANKS)==1) distributes switching power evenly across all SRAM banks"));
    std::printf("    }\n");
    std::printf("  },\n");

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
