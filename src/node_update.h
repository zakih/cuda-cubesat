#pragma once
#include "config.h"
#include <cmath>

// ============================================================================
// The per-node heat-balance update, factored out so the serial and OpenMP
// backends compute *provably identical* arithmetic (and the CUDA kernel mirrors
// it line for line). Keeping one definition is what makes the cross-backend
// correctness check meaningful.
//
// In-panel conduction is a 4-neighbor stencil; nodes on a panel's mesh boundary
// simply have fewer in-panel neighbors (no wraparound). Cross-panel coupling is
// applied separately by the caller via the edge list (so this function handles
// only the dense in-panel stencil + radiation + loads).
// ============================================================================

// Compute the new temperature for in-panel node (p,r,c).
// `T` is the full field; `n` the per-panel mesh size; coefficients precomputed.
static inline real update_node(
    const real* T, int p, int r, int c, int n,
    real inv_C, real g, real sigma_eps_A,
    real q_ext, real q_gen, real dt)
{
    const int base = p * n * n;
    const int k = base + r * n + c;
    const real Ti = T[k];

    // 4-neighbor in-panel conduction (skip neighbors off the mesh edge)
    real cond = real(0);
    if (c > 0)     cond += g * (T[k - 1]  - Ti);
    if (c < n - 1) cond += g * (T[k + 1]  - Ti);
    if (r > 0)     cond += g * (T[k - n]  - Ti);
    if (r < n - 1) cond += g * (T[k + n]  - Ti);

    // radiation to deep space (T^4 term)
    const real Ti2 = Ti * Ti;
    const real rad = sigma_eps_A * (Ti2 * Ti2 - (T_SPACE * T_SPACE * T_SPACE * T_SPACE));

    // forward-Euler
    const real dTdt = (cond - rad + q_ext + q_gen) * inv_C;
    return Ti + dTdt * dt;
}
