#include "solver.h"
#include "node_update.h"
#include <chrono>
#include <cmath>
#include <algorithm>

// One full forward-Euler step over all nodes: in-panel stencil + radiation +
// loads (via update_node), then cross-panel edge conduction added separately.
// Reads `cur`, writes `nxt`. Returns max|ΔT| over all nodes.
static real step_serial(const real* cur, real* nxt, const Model& m,
                        real inv_C, real g, real sigma_eps_A, real dt) {
    const int n = m.geo.n;
    const int P = m.geo.num_panels();

    // dense in-panel update
    for (int p = 0; p < P; ++p)
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                const int k = p * n * n + r * n + c;
                nxt[k] = update_node(cur, p, r, c, n, inv_C, g, sigma_eps_A,
                                     m.q_ext[k], m.q_gen[k], dt);
            }

    // cross-panel edge conduction: add the heat exchanged across each stitched
    // pair to BOTH endpoints (symmetric). We apply it as an explicit increment
    // using the *old* temperatures, consistent with forward-Euler.
    for (const auto& e : m.edges) {
        const real flux = g * (cur[e.b] - cur[e.a]);  // into a from b
        nxt[e.a] += flux * inv_C * dt;
        nxt[e.b] -= flux * inv_C * dt;
    }

    // track convergence
    real maxdT = real(0);
    const int N = m.N;
    for (int k = 0; k < N; ++k)
        maxdT = std::max(maxdT, std::fabs(nxt[k] - cur[k]));
    return maxdT;
}

SolveResult solve_serial(std::vector<real>& T, const Model& m, const RunConfig& cfg, Mode mode) {
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
        real maxdT = step_serial(cur, nxt, m, inv_C, g, sigma_eps_A, dt);
        std::swap(cur, nxt);
        res.steps_run = s + 1;
        res.final_maxdT = maxdT;
        if (mode == Mode::Converge && maxdT < cfg.converge_tol) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ensure result lands in caller's T
    if (cur != T.data()) std::copy(cur, cur + m.N, T.data());
    return res;
}
