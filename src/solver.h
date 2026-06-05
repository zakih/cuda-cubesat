#pragma once
#include "config.h"
#include "geometry.h"
#include <vector>

// Result of a solver run.
struct SolveResult {
    double elapsed_ms = 0.0;  // wall-clock for the integration loop (excludes setup)
    int    steps_run  = 0;    // steps actually executed
    real   final_maxdT = real(0);  // last max|ΔT| (for convergence mode)
};

// Run modes:
//   Timing      : execute exactly cfg.timing_steps steps (fair work across N).
//   Converge    : run until max|ΔT| < cfg.converge_tol or cfg.max_steps.
enum class Mode { Timing, Converge };

// Each backend exposes the same signature. On return, T holds the final field.
SolveResult solve_serial(std::vector<real>& T, const Model& m, const RunConfig& cfg, Mode mode);
SolveResult solve_omp   (std::vector<real>& T, const Model& m, const RunConfig& cfg, Mode mode);

// CUDA variants (defined in solver_cuda.cu). `tiled` selects the shared-memory
// kernel; otherwise the naive global-memory kernel.
SolveResult solve_cuda  (std::vector<real>& T, const Model& m, const RunConfig& cfg, Mode mode, bool tiled);
