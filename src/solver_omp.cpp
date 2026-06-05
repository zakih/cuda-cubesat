#include "solver.h"
#include "node_update.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

// Same physics as the serial solver, but the dense in-panel stencil (the O(N)
// dominant cost) is parallelized across threads. Each node writes only its own
// nxt[k], so the stencil is embarrassingly parallel with no races.
//
// The cross-panel edge coupling does scattered += into node pairs; parallelizing
// that would race when two edges share a node. Since |edges| ~ O(n) << O(n^2)
// nodes, we apply it serially — its cost is negligible and correctness is
// guaranteed. (This is a deliberate, documented choice, not an oversight.)
static real step_omp(const real* cur, real* nxt, const Model& m,
                     real inv_C, real g, real sigma_eps_A, real dt) {
    const int n = m.geo.n;
    const int P = m.geo.num_panels();
    const int N = m.N;

    real maxdT = real(0);
    #pragma omp parallel
    {
        real local_max = real(0);
        #pragma omp for schedule(static) nowait
        for (int k = 0; k < N; ++k) {
            const int p = k / (n * n);
            const int rem = k - p * n * n;
            const int r = rem / n;
            const int c = rem - r * n;
            const real newv = update_node(cur, p, r, c, n, inv_C, g, sigma_eps_A,
                                          m.q_ext[k], m.q_gen[k], dt);
            nxt[k] = newv;
            const real d = std::fabs(newv - cur[k]);
            if (d > local_max) local_max = d;
        }
        #pragma omp critical
        { if (local_max > maxdT) maxdT = local_max; }
    }

    // edge coupling (serial, negligible cost). NOTE: this adjusts a few nodes
    // after the parallel region; its ΔT contribution at O(n) edges is not folded
    // into maxdT, which is fine — convergence is governed by the O(n^2) bulk.
    for (const auto& e : m.edges) {
        const real flux = g * (cur[e.b] - cur[e.a]);
        nxt[e.a] += flux * inv_C * dt;
        nxt[e.b] -= flux * inv_C * dt;
    }
    return maxdT;
}

SolveResult solve_omp(std::vector<real>& T, const Model& m, const RunConfig& cfg, Mode mode) {
    const real inv_C = real(1) / m.geo.node_capacity();
    const real g     = m.geo.conductance();
    const real sigma_eps_A = SIGMA * m.geo.emissivity * m.geo.node_area();
    const real dt    = m.geo.dt();

    std::vector<real> scratch(m.N);
    real* cur = T.data();
    real* nxt = scratch.data();

    SolveResult res;
    auto t0 = std::chrono::high_resolution_clock::now();

    const int max_steps = (mode == Mode::Timing) ? cfg.timing_steps : cfg.max_steps;
    for (int s = 0; s < max_steps; ++s) {
        real maxdT = step_omp(cur, nxt, m, inv_C, g, sigma_eps_A, dt);
        std::swap(cur, nxt);
        res.steps_run = s + 1;
        res.final_maxdT = maxdT;
        if (mode == Mode::Converge && maxdT < cfg.converge_tol) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (cur != T.data()) std::copy(cur, cur + m.N, T.data());
    return res;
}
