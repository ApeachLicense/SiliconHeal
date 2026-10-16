#include "hw_emulator_harness.hpp"

// Kernel entry point implemented in kernel.cpp
void run_hx9000_tensor_kernel(const float* dram_in, float* dram_out, float alpha);

// ============================================================================
// Solve 2D Steady-State Heat Diffusion PDE on 8x8 Silicon Die Grid (Gauss-Seidel)
// Driven continuously by observed hardware counters (bank_access_hist & bank_conflict_hist)
// ============================================================================
static void solve_die_thermal_pde(bool is_bus_fault, uint64_t total_colliding_lanes, uint64_t total_gather_lanes, bool burst16_on) {
    const double t_ambient = 45.0;
    double power_source[8][8];

    // Base logic & clock distribution power across the 8x8 die grid
    const double base_logic_power = burst16_on ? 1.8 : 2.9;
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            power_source[r][c] = base_logic_power;
            g_hw.die_thermal_grid[r][c] = t_ambient + 10.0;
        }
    }

    if (is_bus_fault) {
        // AXI4 Arbiter & DMA Controller macro lockup (top-left die quadrant)
        g_hw.bank0_power_density_w_mm2 = 46.8;
        for (int r = 1; r < 5; ++r) {
            for (int c = 1; c < 5; ++c) {
                power_source[r][c] = 28.5;
            }
        }
        power_source[2][2] = 46.8;
        power_source[2][3] = 44.0;
    } else {
        // Continuous power density model driven by actual observed SRAM access & collision counters
        const double collision_ratio = (total_gather_lanes > 0)
            ? (static_cast<double>(total_colliding_lanes) / static_cast<double>(total_gather_lanes))
            : 0.0;
        // Normalized collision intensity (1.0 when (N-1)/N lanes collide)
        const double max_possible_ratio = static_cast<double>(NUM_BANKS - 1) / static_cast<double>(NUM_BANKS);
        const double stall_intensity = (max_possible_ratio > 0.0) ? (collision_ratio / max_possible_ratio) : 0.0;

        // Distributed dynamic SRAM bank access power + localized crossbar/sense-amp stall heating
        const double uniform_bank_power = 7.94;
        const double stall_hotspot_peak = 38.0 * stall_intensity;
        g_hw.bank0_power_density_w_mm2 = uniform_bank_power + stall_hotspot_peak;

        for (int r = 1; r < 7; ++r) {
            for (int c = 1; c < 7; ++c) {
                power_source[r][c] = uniform_bank_power;
            }
        }

        if (stall_intensity > 0.001) {
            for (int r = 1; r < 7; ++r) {
                power_source[r][2] += 15.8 * stall_intensity;
                power_source[r][3] += 29.1 * stall_intensity;
                power_source[r][4] += 17.6 * stall_intensity;
            }
            power_source[3][3] = uniform_bank_power + stall_hotspot_peak;
            power_source[4][3] = uniform_bank_power + (stall_hotspot_peak * 0.97);
        }
    }

    // 80 iterations of 2D Finite-Difference Gauss-Seidel relaxation:
    // T[r][c] = (T_neighbors + P[r][c] + h_sink * T_amb) / (4 + h_sink)
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
    hw_flush_pending_wavefront();

    // ========================================================================
    // Anti-Cheat Hardware Verification:
    // Every valid kernel MUST stage all 65,536 elements through SramTile<T, R, C>
    // and perform column-strided transpose reads/writes across the tiles.
    // Raw uninstrumented C arrays (`float local_tile[8][8]`) or omitted accesses
    // are immediately flagged as ERR_SRAM_TILE_BYPASS_DETECTED.
    // ========================================================================
    const uint64_t expected_elements = static_cast<uint64_t>(MATRIX_DIM) * static_cast<uint64_t>(MATRIX_DIM);
    const uint64_t expected_transpose_groups = expected_elements / static_cast<uint64_t>(TILE_DIM);

    if (!g_hw.fault_triggered) {
        if (g_hw.sram_word_writes < expected_elements || g_hw.sram_word_reads < expected_elements) {
            g_hw.fault_triggered = true;
            g_hw.sram_bypass_detected = true;
            std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_SRAM_TILE_BYPASS_DETECTED");
            std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
                "Hardware Verification Fault: Kernel bypassed instrumented SramTile<T, Rows, Cols> (observed writes=%llu, reads=%llu, expected=%llu). Raw uninstrumented C arrays are forbidden.",
                (unsigned long long)g_hw.sram_word_writes,
                (unsigned long long)g_hw.sram_word_reads,
                (unsigned long long)expected_elements);
        } else if (g_hw.strided_transpose_groups < expected_transpose_groups) {
            g_hw.fault_triggered = true;
            g_hw.sram_bypass_detected = true;
            std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_SRAM_TRANSPOSE_BYPASS_DETECTED");
            std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
                "Hardware Verification Fault: Kernel did not perform column-strided transpose gather/scatter through SramTile (observed strided groups=%llu, expected>=%llu).",
                (unsigned long long)g_hw.strided_transpose_groups,
                (unsigned long long)expected_transpose_groups);
        }
    }

    // Verify functional correctness against Golden Reference (CRC32)
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
            "Sec 5.1 & Errata #E-402 Violation: Missing hw_dmb() barrier and non-volatile DMA_SR polling left %d dirty L1 cache lines unflushed.",
            g_hw.dirty_l1_lines_bypassed);
    } else if (!g_hw.fault_triggered && matched_elements < total_elements) {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "ERR_CRC_MISMATCH_STALE_CACHE");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "Sec 5.1.1 & Errata #E-402 Violation: Missing hw_dmb() before DMA trigger resulted in stale DRAM read.");
    }

    bool crc_passed = (!g_hw.fault_triggered) && (matched_elements == total_elements) && g_hw.volatile_status_polled;
    double accuracy_pct = g_hw.fault_triggered ? 0.0 : (100.0 * matched_elements / static_cast<double>(total_elements));

    // Compute Bank Conflict Rate directly from observed bank collision histogram on gather reads
    uint64_t total_colliding_lanes = 0;
    for (int b = 0; b < NUM_BANKS; ++b) {
        total_colliding_lanes += g_hw.bank_conflict_hist[b];
    }
    const uint64_t total_gather_lanes = g_hw.read_sram_groups * static_cast<uint64_t>(NUM_BANKS);
    double bank_conflict_rate = (!g_hw.fault_triggered && total_gather_lanes > 0)
        ? (100.0 * static_cast<double>(total_colliding_lanes) / static_cast<double>(total_gather_lanes))
        : 0.0;

    // 9-Layer Inference Pipeline Cycle Model @ 400 MHz Core Clock
    uint64_t weighted_layer_cycles = g_hw.dma_cycles + g_hw.sram_compute_cycles;
    uint64_t frame_cycles = g_hw.fault_triggered ? 0ULL : (weighted_layer_cycles * 9ULL);
    double latency_ms = g_hw.fault_triggered ? 0.0 : (static_cast<double>(frame_cycles) / 400000.0);
    double throughput_gops = (latency_ms > 0.0) ? (67.108864 / latency_ms) : 0.0;

    bool burst16_on = ((g_hw.DMA_CR & DMA_CR_BURST16) != 0);
    solve_die_thermal_pde(g_hw.fault_triggered, total_colliding_lanes, total_gather_lanes, burst16_on);

    // ========================================================================
    // Analytical PDN Voltage & Real-Time Jitter/WCET Models Driven by Counters
    // (Explicitly marked as counter-driven analytical models in JSON output)
    // ========================================================================
    const double unburst_beats = burst16_on ? 0.0 : 32768.0;
    const double conflict_groups = static_cast<double>(g_hw.conflicted_sram_groups);
    const double dirty_lines = static_cast<double>(g_hw.dirty_l1_lines_bypassed);

    if (g_hw.fault_triggered) {
        if (g_hw.DMA_SR & DMA_SR_ERR_AXI_DEADLOCK) {
            g_hw.pdn_di_dt_spike_a_per_ns = 18.4;
        } else if (g_hw.DMA_SR & DMA_SR_ERR_UNALIGNED) {
            g_hw.pdn_di_dt_spike_a_per_ns = 14.5;
        } else {
            g_hw.pdn_di_dt_spike_a_per_ns = 12.0;
        }
        g_hw.vdd_droop_mv = 850.0 - 9.1 * g_hw.pdn_di_dt_spike_a_per_ns;
    } else {
        // Continuous analytical PDN di/dt equation from bus & cache activity
        g_hw.pdn_di_dt_spike_a_per_ns =
            1.1
            + (unburst_beats / 32768.0) * 1.3
            + (dirty_lines / 56.0) * 1.5
            + (conflict_groups / 8192.0) * 0.4;
        g_hw.vdd_droop_mv = 850.0 - 8.0 * g_hw.pdn_di_dt_spike_a_per_ns;
    }

    double jitter_us = 0.0;
    double wcet_ms = 0.0;
    if (g_hw.fault_triggered) {
        jitter_us = 9999.0; // Bus lockup / halt
        wcet_ms = 0.0;
    } else {
        // Continuous jitter model from single-beat bus re-arbitrations, bank conflict stalls, and L1 snoop retries
        jitter_us = 2.5
                    + 0.0035 * unburst_beats
                    + 0.0210 * conflict_groups
                    + 2.2000 * dirty_lines;
        wcet_ms = latency_ms * (1.02 + 0.08 * (conflict_groups / 8192.0) + 0.04 * (unburst_beats / 32768.0))
                  + (3.0 * jitter_us) / 1000.0;
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
            "Sec 6.4.2 & Errata #E-403 Bottleneck: SramTile<float, %d, %d> stride causes %.2f%% %d-Bank gather conflict & %.1fC Bank 0 hotspot (Latency %.2fms, WCET %.2fms > %.2fms SLA).",
            g_hw.observed_tile_rows, g_hw.observed_tile_cols, bank_conflict_rate, NUM_BANKS, g_hw.peak_die_temp_c, latency_ms, wcet_ms, static_cast<double>(HW_SLA_TARGET_MS));
    } else {
        std::snprintf(g_hw.fault_code, sizeof(g_hw.fault_code), "OK_SPEC_COMPLIANT");
        std::snprintf(g_hw.fault_detail, sizeof(g_hw.fault_detail),
            "All HX-9000 specs verified (Align=%dB, Bit %d clear, DMB+Volatile, BURST_16=1, SramTile<%d,%d> -> 0%% conflict, Latency=%.2fms, WCET=%.2fms, Jitter=%.1fus, PeakTemp=%.1fC).",
            HW_DMA_ALIGN_BYTES, HW_RSVD_MBZ_BIT, g_hw.observed_tile_rows, g_hw.observed_tile_cols, latency_ms, wcet_ms, jitter_us, g_hw.peak_die_temp_c);
    }

    const uintptr_t align_mask = static_cast<uintptr_t>(HW_DMA_ALIGN_BYTES - 1);
    const uintptr_t dst_offset_bytes = (g_hw.registered_base_addr != 0)
        ? (g_hw.DMA_DST_ADDR - g_hw.registered_base_addr)
        : (g_hw.DMA_DST_ADDR & align_mask);
    bool dst_aligned_ok = ((dst_offset_bytes & align_mask) == 0) &&
                          (g_hw.declared_dst_align >= static_cast<size_t>(HW_DMA_ALIGN_BYTES));

    int wavefront_ways = g_hw.fault_triggered ? 0 : g_hw.max_way_conflict_observed;

    // Emit JSON Telemetry
    std::printf("{\n");
    std::printf("  \"status\": \"%s\",\n", overall_status);
    std::printf("  \"fault_code\": \"%s\",\n", g_hw.fault_code);
    std::printf("  \"fault_detail\": \"%s\",\n", g_hw.fault_detail);
    std::printf("  \"telemetry_provenance\": {\n");
    std::printf("    \"sram_and_crc_verification\": \"instrumented_sram_tile_proxy_and_fnv1a_crc32\",\n");
    std::printf("    \"cycle_model\": \"mechanistic_axi4_beats_and_crossbar_serialization_stalls\",\n");
    std::printf("    \"thermal_and_vdd_model\": \"counter_driven_2d_fd_pde_and_analytical_pdn_model\"\n");
    std::printf("  },\n");
    std::printf("  \"spec_config\": {\"dma_align_bytes\": %d, \"rsvd_mbz_bit\": %d, \"num_banks\": %d, \"sla_target_ms\": %.2f},\n",
        HW_DMA_ALIGN_BYTES, HW_RSVD_MBZ_BIT, NUM_BANKS, static_cast<double>(HW_SLA_TARGET_MS));
    std::printf("  \"dma_cr_hex\": \"0x%02X\",\n", g_hw.DMA_CR);
    std::printf("  \"declared_dst_align\": %zu,\n", g_hw.declared_dst_align);
    std::printf("  \"observed_tile_rows\": %d,\n", g_hw.observed_tile_rows);
    std::printf("  \"observed_tile_cols\": %d,\n", g_hw.observed_tile_cols);
    std::printf("  \"sram_word_writes\": %llu,\n", (unsigned long long)g_hw.sram_word_writes);
    std::printf("  \"sram_word_reads\": %llu,\n", (unsigned long long)g_hw.sram_word_reads);
    std::printf("  \"strided_transpose_groups\": %llu,\n", (unsigned long long)g_hw.strided_transpose_groups);
    std::printf("  \"sram_bypass_detected\": %s,\n", g_hw.sram_bypass_detected ? "true" : "false");
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

    // Counter-Driven Physical Models (PDN Voltage Droop vs. 2D Thermal PDE Hotspot)
    std::printf("  \"causal_graph\": {\n");
    std::printf("    \"model_type\": \"counter_driven_analytical_pdn_and_2d_fd_pde\",\n");
    std::printf("    \"voltage_droop_cause\": {\n");
    std::printf("      \"phenomenon\": \"VDD_CORE = %.1f mV (Nominal 850.0 mV, Analytical PDN Model)\",\n", g_hw.vdd_droop_mv);
    std::printf("      \"physical_mechanism\": \"PDN Inductive Drop (V_droop = L_pkg * di/dt + I * R_pdn)\",\n");
    std::printf("      \"di_dt_spike_a_per_ns\": %.1f,\n", g_hw.pdn_di_dt_spike_a_per_ns);
    std::printf("      \"root_trigger\": \"%s\"\n",
        g_hw.fault_triggered
            ? "Unaligned DMA address trap or DMA_CR[RSVD_MBZ] bus deadlock causing AXI4 arbiter retry current spikes (Errata #E-401)"
            : "Clean aligned DMA transfer with zero arbiter deadlock oscillation");
    std::printf("    },\n");
    std::printf("    \"thermal_hotspot_cause\": {\n");
    std::printf("      \"phenomenon\": \"Peak Die Temp = %.1f C (Avg %.1f C, 8x8 FD-PDE Solver)\",\n", g_hw.peak_die_temp_c, g_hw.avg_die_temp_c);
    std::printf("      \"physical_mechanism\": \"2D Steady-State Heat Conduction driven by bank_conflict_hist[]\",\n");
    std::printf("      \"bank0_power_density_w_mm2\": %.2f,\n", g_hw.bank0_power_density_w_mm2);
    std::printf("      \"wavefront_serialization_ways\": %d,\n", wavefront_ways);
    std::printf("      \"root_trigger\": \"%s\"\n",
        (bank_conflict_rate > 20.0)
            ? "Unpadded SramTile<T, R, C> stride routes column-gather vector lanes to the same bank, causing N-way wavefront serialization stalls (Errata #E-403)"
            : (g_hw.fault_triggered
                ? "AXI4 arbiter bus lock localized current crowding in DMA controller macro"
                : "Coprime padded SramTile stride (gcd(Cols, NUM_BANKS)==1) distributes switching power evenly across all SRAM banks"));
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
