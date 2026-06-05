// ============================================================================
// CubeSat thermal simulator — driver
//
//   ./cubesat_thermal sweep   [out.csv]   -> node-count sweep, all backends,
//                                            timing mode, writes CSV
//   ./cubesat_thermal field   [n] [out]   -> run to steady state at resolution n,
//                                            dump temperature field (binary)
//
// The sweep uses a FIXED step count per size so every backend does identical
// work (fair comparison). The field dump runs to convergence so the dumped
// temperatures are a true steady state.
// ============================================================================

#include "config.h"
#include "geometry.h"
#include "solver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

#ifdef HAVE_CUDA
static const bool kHaveCuda = true;
#else
static const bool kHaveCuda = false;
// stub so the program links CPU-only when built without CUDA
SolveResult solve_cuda(std::vector<real>&, const Model&, const RunConfig&, Mode, bool) {
    SolveResult r; r.elapsed_ms = -1.0; return r;  // -1 => not available
}
#endif

static void run_sweep(const std::string& csv_path) {
    // n values: 8,16,...,512  -> N = 6*n^2 from 384 to ~1.5M
    std::vector<int> ns = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
    RunConfig cfg;
    cfg.timing_steps = 100;  // fixed work per size

    FILE* f = std::fopen(csv_path.c_str(), "w");
    std::fprintf(f, "n,N,serial_ms,openmp_ms,cuda_naive_ms,cuda_tiled_ms,max_abs_diff\n");
    std::printf("%-6s %-10s %-12s %-12s %-14s %-14s %-12s\n",
                "n", "N", "serial", "openmp", "cuda_naive", "cuda_tiled", "max|diff|");

    for (int n : ns) {
        Geometry g; g.n = n;
        Model m(g);

        std::vector<real> T_ref, T;
        // serial (reference)
        m.init_field(T_ref);
        SolveResult rs = solve_serial(T_ref, m, cfg, Mode::Timing);

        // openmp
        m.init_field(T);
        SolveResult ro = solve_omp(T, m, cfg, Mode::Timing);
        real diff_omp = 0;
        for (int i = 0; i < m.N; ++i) diff_omp = std::fmax(diff_omp, std::fabs(T[i] - T_ref[i]));

        // cuda naive + tiled
        double cn = -1, ct = -1; real diff_cuda = 0;
        if (kHaveCuda) {
            m.init_field(T);
            SolveResult rcn = solve_cuda(T, m, cfg, Mode::Timing, /*tiled=*/false);
            cn = rcn.elapsed_ms;
            for (int i = 0; i < m.N; ++i) diff_cuda = std::fmax(diff_cuda, std::fabs(T[i] - T_ref[i]));

            m.init_field(T);
            SolveResult rct = solve_cuda(T, m, cfg, Mode::Timing, /*tiled=*/true);
            ct = rct.elapsed_ms;
        }

        real max_diff = std::fmax(diff_omp, diff_cuda);
        std::fprintf(f, "%d,%d,%.3f,%.3f,%.3f,%.3f,%.3e\n",
                     n, m.N, rs.elapsed_ms, ro.elapsed_ms, cn, ct, max_diff);
        std::fflush(f);
        std::printf("%-6d %-10d %-12.2f %-12.2f %-14.2f %-14.2f %-12.2e\n",
                    n, m.N, rs.elapsed_ms, ro.elapsed_ms, cn, ct, max_diff);
    }
    std::fclose(f);
    std::printf("\nSweep written to %s\n", csv_path.c_str());
}

static void run_field(int n, const std::string& out_path) {
    Geometry g; g.n = n;
    Model m(g);
    RunConfig cfg;
    cfg.converge_tol = real(1e-4);

    // Prefer CUDA tiled for the field solve if available (fastest), else serial.
    std::vector<real> T; m.init_field(T);
    SolveResult r;
    const char* used;
    if (kHaveCuda) { r = solve_cuda(T, m, cfg, Mode::Converge, true); used = "cuda_tiled"; }
    else           { r = solve_serial(T, m, cfg, Mode::Converge);    used = "serial"; }

    std::printf("field solve (%s): n=%d N=%d, %d steps, %.1f ms, final max|dT|=%.2e\n",
                used, n, m.N, r.steps_run, r.elapsed_ms, r.final_maxdT);

    // dump: magic | n | P | N | float32[N]  (panel-major, row-major within panel)
    FILE* f = std::fopen(out_path.c_str(), "wb");
    std::fwrite("CSAT", 1, 4, f);
    int hdr[3] = { n, m.geo.num_panels(), m.N };
    std::fwrite(hdr, sizeof(int), 3, f);
    // write as float32 regardless of `real` so the Python reader is fixed-format
    std::vector<float> buf(m.N);
    for (int i = 0; i < m.N; ++i) buf[i] = (float)T[i];
    std::fwrite(buf.data(), sizeof(float), m.N, f);
    std::fclose(f);
    std::printf("temperature field written to %s\n", out_path.c_str());
}

int main(int argc, char** argv) {
    std::string cmd = (argc > 1) ? argv[1] : "sweep";
    if (cmd == "sweep") {
        std::string out = (argc > 2) ? argv[2] : "output/sweep.csv";
        run_sweep(out);
    } else if (cmd == "field") {
        int n = (argc > 2) ? std::atoi(argv[2]) : 128;
        std::string out = (argc > 3) ? argv[3] : "output/field.bin";
        run_field(n, out);
    } else {
        std::fprintf(stderr, "usage: %s [sweep out.csv | field n out.bin]\n", argv[0]);
        return 1;
    }
    return 0;
}
