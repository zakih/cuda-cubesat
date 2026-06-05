#include "solver.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// CUDA backends. Both kernels compute the SAME per-node heat balance as the CPU
// update_node() (in-panel 4-neighbour conduction + radiation + loads). The
// cross-panel edge coupling is a small O(n) scatter applied by a separate tiny
// kernel using atomics, so it never races.
//
//   naive : every neighbour read goes to global memory.
//   tiled : each block stages its panel tile + 1-cell halo into shared memory,
//           so the 4 neighbour reads per node come from fast on-chip memory.
//           This is meaningful here precisely because each panel is a
//           contiguous n×n grid (chosen in geometry.h for exactly this reason).
// ============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e__ = (call);                                              \
        if (e__ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(e__));                             \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static constexpr int TILE = 16;
static const real T_SPACE4 = T_SPACE * T_SPACE * T_SPACE * T_SPACE;

// ---- naive kernel: one thread per node, neighbours from global memory ------
__global__ void k_naive(const real* __restrict__ cur, real* __restrict__ nxt,
                        const real* __restrict__ q_ext, const real* __restrict__ q_gen,
                        int n, int P, real inv_C, real g, real sigma_eps_A,
                        real dt, real t_space4) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = P * n * n;
    if (gid >= N) return;

    int p = gid / (n * n);
    int rem = gid - p * n * n;
    int r = rem / n;
    int c = rem - r * n;
    int base = p * n * n;
    int k = base + r * n + c;
    real Ti = cur[k];

    real cond = real(0);
    if (c > 0)     cond += g * (cur[k - 1] - Ti);
    if (c < n - 1) cond += g * (cur[k + 1] - Ti);
    if (r > 0)     cond += g * (cur[k - n] - Ti);
    if (r < n - 1) cond += g * (cur[k + n] - Ti);

    real Ti2 = Ti * Ti;
    real rad = sigma_eps_A * (Ti2 * Ti2 - t_space4);
    nxt[k] = Ti + (cond - rad + q_ext[k] + q_gen[k]) * inv_C * dt;
}

// ---- tiled kernel: shared-memory staging per panel block -------------------
// Grid is 3D: (tiles_x, tiles_y, panels). Each block handles a TILE×TILE patch
// of one panel. Threads load their cell + halo ring into shared memory, sync,
// then read neighbours from shared memory.
__global__ void k_tiled(const real* __restrict__ cur, real* __restrict__ nxt,
                        const real* __restrict__ q_ext, const real* __restrict__ q_gen,
                        int n, real inv_C, real g, real sigma_eps_A,
                        real dt, real t_space4) {
    __shared__ real s[TILE + 2][TILE + 2];

    int p  = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;
    int c  = blockIdx.x * TILE + tx;   // col within panel
    int r  = blockIdx.y * TILE + ty;   // row within panel
    int sx = tx + 1, sy = ty + 1;
    int base = p * n * n;

    // center load (guarded against partial edge blocks)
    if (r < n && c < n) s[sy][sx] = cur[base + r * n + c];

    // halo ring (only valid where a real neighbour exists inside the panel)
    if (tx == 0 && c > 0)             s[sy][0]        = cur[base + r * n + (c - 1)];
    if (tx == TILE - 1 && c < n - 1)  s[sy][TILE + 1] = cur[base + r * n + (c + 1)];
    if (ty == 0 && r > 0)             s[0][sx]        = cur[base + (r - 1) * n + c];
    if (ty == TILE - 1 && r < n - 1)  s[TILE + 1][sx] = cur[base + (r + 1) * n + c];

    __syncthreads();

    if (r >= n || c >= n) return;
    int k = base + r * n + c;
    real Ti = s[sy][sx];

    real cond = real(0);
    if (c > 0)     cond += g * (s[sy][sx - 1] - Ti);
    if (c < n - 1) cond += g * (s[sy][sx + 1] - Ti);
    if (r > 0)     cond += g * (s[sy - 1][sx] - Ti);
    if (r < n - 1) cond += g * (s[sy + 1][sx] - Ti);

    real Ti2 = Ti * Ti;
    real rad = sigma_eps_A * (Ti2 * Ti2 - t_space4);
    nxt[k] = Ti + (cond - rad + q_ext[k] + q_gen[k]) * inv_C * dt;
}

// ---- edge-coupling kernel: scattered cross-panel conduction via atomics ----
__global__ void k_edges(const real* __restrict__ cur, real* __restrict__ nxt,
                        const int* __restrict__ ea, const int* __restrict__ eb,
                        int num_edges, real inv_C, real g, real dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    int a = ea[i], b = eb[i];
    real flux = g * (cur[b] - cur[a]) * inv_C * dt;
    atomicAdd(&nxt[a],  flux);
    atomicAdd(&nxt[b], -flux);
}

// ---- max|ΔT| reduction kernel (block-wise, then host finishes) -------------
__global__ void k_maxdiff(const real* __restrict__ a, const real* __restrict__ b,
                         int N, real* __restrict__ blockmax) {
    extern __shared__ real sm[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    real v = real(0);
    if (i < N) v = fabsf(a[i] - b[i]);
    sm[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    if (tid == 0) blockmax[blockIdx.x] = sm[0];
}

SolveResult solve_cuda(std::vector<real>& T, const Model& m, const RunConfig& cfg,
                       Mode mode, bool tiled) {
    const int n = m.geo.n, P = m.geo.num_panels(), N = m.N;
    const size_t bytes = (size_t)N * sizeof(real);
    const real inv_C = real(1) / m.geo.node_capacity();
    const real g     = m.geo.conductance();
    const real sigma_eps_A = SIGMA * m.geo.emissivity * m.geo.node_area();
    const real dt    = m.geo.dt();

    // device buffers
    real *d_cur, *d_nxt, *d_qext, *d_qgen;
    CUDA_CHECK(cudaMalloc(&d_cur,  bytes));
    CUDA_CHECK(cudaMalloc(&d_nxt,  bytes));
    CUDA_CHECK(cudaMalloc(&d_qext, bytes));
    CUDA_CHECK(cudaMalloc(&d_qgen, bytes));
    CUDA_CHECK(cudaMemcpy(d_cur,  T.data(),     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nxt,  T.data(),     bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qext, m.q_ext.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qgen, m.q_gen.data(), bytes, cudaMemcpyHostToDevice));

    // edges to device (SoA)
    std::vector<int> ea(m.edges.size()), eb(m.edges.size());
    for (size_t i = 0; i < m.edges.size(); ++i) { ea[i] = m.edges[i].a; eb[i] = m.edges[i].b; }
    int *d_ea = nullptr, *d_eb = nullptr;
    int num_edges = (int)m.edges.size();
    if (num_edges > 0) {
        CUDA_CHECK(cudaMalloc(&d_ea, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_eb, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ea, ea.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_eb, eb.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    }

    // launch configs
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    dim3 tblock(TILE, TILE);
    dim3 tgrid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE, P);
    int eblocks = (num_edges + threads - 1) / threads;

    // reduction scratch
    real* d_blockmax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockmax, blocks * sizeof(real)));
    std::vector<real> h_blockmax(blocks);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    SolveResult res;
    const int max_steps = (mode == Mode::Timing) ? cfg.timing_steps : cfg.max_steps;
    const bool check_conv = (mode == Mode::Converge);

    CUDA_CHECK(cudaEventRecord(t0));
    for (int s = 0; s < max_steps; ++s) {
        if (tiled)
            k_tiled<<<tgrid, tblock>>>(d_cur, d_nxt, d_qext, d_qgen, n,
                                       inv_C, g, sigma_eps_A, dt, T_SPACE4);
        else
            k_naive<<<blocks, threads>>>(d_cur, d_nxt, d_qext, d_qgen, n, P,
                                         inv_C, g, sigma_eps_A, dt, T_SPACE4);
        if (num_edges > 0)
            k_edges<<<eblocks, threads>>>(d_cur, d_nxt, d_ea, d_eb,
                                          num_edges, inv_C, g, dt);

        real maxdT = real(0);
        if (check_conv) {
            k_maxdiff<<<blocks, threads, threads * sizeof(real)>>>(d_nxt, d_cur, N, d_blockmax);
            CUDA_CHECK(cudaMemcpy(h_blockmax.data(), d_blockmax, blocks * sizeof(real),
                                  cudaMemcpyDeviceToHost));
            for (int b = 0; b < blocks; ++b) maxdT = std::max(maxdT, h_blockmax[b]);
        }

        std::swap(d_cur, d_nxt);
        res.steps_run = s + 1;
        res.final_maxdT = maxdT;
        if (check_conv && maxdT < cfg.converge_tol) break;
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    res.elapsed_ms = ms;

    CUDA_CHECK(cudaMemcpy(T.data(), d_cur, bytes, cudaMemcpyDeviceToHost));

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_cur); cudaFree(d_nxt); cudaFree(d_qext); cudaFree(d_qgen);
    cudaFree(d_blockmax);
    if (d_ea) cudaFree(d_ea);
    if (d_eb) cudaFree(d_eb);
    return res;
}
