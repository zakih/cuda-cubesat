#pragma once
// ============================================================================
// CubeSat thermal simulator — global configuration
// ============================================================================
// A 6U CubeSat (≈10×20×30 cm) modeled as 6 panels, each meshed into an n×n grid
// of thermal sub-nodes. Per node, per timestep, we integrate the heat balance:
//
//   C dT/dt =  Σ_neighbors k_ij (T_j - T_i)        [conduction, linear in T]
//            - σ ε A (T_i^4 - T_space^4)           [radiation to deep space, T^4]
//            + Q_solar (sunlit external faces)      [external load]
//            + Q_gen   (component nodes)            [internal generation]
//
// Integrated with explicit forward-Euler. The conduction term sets a hard
// stability limit on dt; we derive dt from the mesh so the sim is always stable
// as the mesh is refined (which is exactly what the node-count sweep does).
// ============================================================================

#include <string>

// ---- precision -------------------------------------------------------------
// float32 by default: ~2x throughput on Pascal (GTX 1050) vs float64, and more
// headroom in 4 GB. Flip this typedef to double for an FP64 build.
using real = float;

// ---- physical constants ----------------------------------------------------
constexpr real SIGMA   = real(5.670374419e-8);  // Stefan–Boltzmann, W/m^2/K^4
constexpr real T_SPACE = real(3.0);             // deep-space sink temperature, K

// ---- CubeSat geometry ------------------------------------------------------
// A 6U is 6 faces. We treat each face as a square panel meshed n×n for
// simplicity (the absolute panel size mostly rescales conduction coefficients;
// the parallelization story is unaffected). N_total ≈ 6 * n^2.
struct Geometry {
    int   n          = 64;          // per-panel mesh resolution (n×n)
    real  panel_size = real(0.20);  // panel edge length, m (representative 6U face)
    real  thickness  = real(0.002); // panel thickness, m (aluminum skin)

    // material: aluminum 6061-ish
    real  k_cond     = real(167.0); // thermal conductivity, W/m/K
    real  rho        = real(2700.0);// density, kg/m^3
    real  cp         = real(896.0); // specific heat, J/kg/K
    real  emissivity = real(0.85);  // surface emissivity (anodized/painted)

    // loads
    real  solar_flux = real(1361.0);// solar constant, W/m^2 (on sunlit faces)
    real  absorptivity = real(0.4); // solar absorptivity of the surface
    real  q_gen_total = real(20.0); // total internal electronics dissipation, W

    int   num_panels() const { return 6; }
    int   nodes_per_panel() const { return n * n; }
    int   total_nodes() const { return num_panels() * nodes_per_panel(); }

    // cell spacing within a panel
    real dx() const { return panel_size / real(n - 1); }

    // Per-node heat capacity C = rho * cp * (cell area * thickness).  [J/K]
    real node_capacity() const {
        real cell = dx() * dx() * thickness;
        return rho * cp * cell;
    }

    // Conductance between adjacent in-panel nodes: g = k * (cross-section)/dx. [W/K]
    // cross-section between two cells = thickness * dx.
    real conductance() const {
        return k_cond * (thickness * dx()) / dx();  // = k_cond * thickness
    }

    // Radiating area per surface node = one cell face. [m^2]
    real node_area() const { return dx() * dx(); }

    // Stable forward-Euler dt. The tightest constraint is a node losing heat to
    // up to 4 conduction neighbors: dt < C / (4 g). We also keep a margin for
    // the (smaller, but nonlinear) radiation term. Use 40% of the conduction
    // limit — comfortably stable in practice (verified in tests).
    real dt() const {
        real C = node_capacity();
        real g = conductance();
        return real(0.4) * C / (real(4.0) * g);
    }
};

// ---- run configuration -----------------------------------------------------
struct RunConfig {
    int  timing_steps = 200;     // fixed step count for fair timing comparison
    real converge_tol = real(1e-3); // max|ΔT| per step to declare steady state
    int  max_steps    = 2000000; // safety cap for convergence mode
    std::string outdir = "output";
};

// Which compute backend.
enum class Backend { Serial, OpenMP, CudaNaive, CudaTiled };

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::Serial:    return "serial";
        case Backend::OpenMP:    return "openmp";
        case Backend::CudaNaive: return "cuda_naive";
        case Backend::CudaTiled: return "cuda_tiled";
    }
    return "?";
}
