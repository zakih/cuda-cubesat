#pragma once
#include "config.h"
#include <vector>

// ============================================================================
// Node layout and per-node static data.
//
// Memory layout: panels are stored back-to-back, each a contiguous n×n
// row-major block. Node index of (panel p, row r, col c):
//     idx = p*n*n + r*n + c
// This contiguous-per-panel layout is what lets the shared-memory tiled CUDA
// kernel treat each panel as a clean 2D grid to tile (a free-form node graph
// could not be tiled).
//
// Conduction:
//   * within a panel: 4-neighbor stencil (handled implicitly by the solvers
//     using the row/col structure)
//   * between panels: panel edges are thermally joined. We precompute a small
//     list of cross-panel node pairs (the shared edges of the cube) and add a
//     conduction term across each pair. This is a sparse "edge coupling" set,
//     separate from the dense in-panel stencil.
//
// Loads (precomputed per node):
//   * q_ext[i]  : external solar input on sunlit faces (W), 0 elsewhere
//   * q_gen[i]  : internal generation (W), nonzero only at component nodes
//   * is_surface: every node is an external radiator on a thin-shell CubeSat,
//     so all nodes radiate. (Kept as a flag for clarity / future internal nodes.)
// ============================================================================

struct EdgeCouple {
    int a;       // node index in one panel
    int b;       // node index in adjacent panel
};

struct Model {
    Geometry geo;

    std::vector<real> q_ext;   // external (solar) load per node, W
    std::vector<real> q_gen;   // internal generation per node, W
    std::vector<EdgeCouple> edges;  // cross-panel conduction pairs

    int N = 0;

    explicit Model(const Geometry& g) : geo(g) { build(); }

    static inline int idx(int p, int r, int c, int n) { return p * n * n + r * n + c; }

    void build() {
        const int n = geo.n;
        N = geo.total_nodes();
        q_ext.assign(N, real(0));
        q_gen.assign(N, real(0));
        edges.clear();

        // ---- external solar load --------------------------------------------
        // Model a simple sun direction: panels 0 and 1 are sunlit ("+X" and
        // "+Y" faces), the rest are in shadow. Per-node solar input =
        // absorptivity * solar_flux * cell_area.
        const real q_solar_node = geo.absorptivity * geo.solar_flux * geo.node_area();
        for (int p = 0; p < 2; ++p)
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c)
                    q_ext[idx(p, r, c, n)] = q_solar_node;

        // ---- internal generation --------------------------------------------
        // Lump the electronics dissipation into a small patch of nodes on the
        // interior of panel 5 (a "circuit board" mounted to one face). Spread
        // q_gen_total evenly over a center block.
        const int lo = n / 2 - n / 8, hi = n / 2 + n / 8;
        int count = 0;
        for (int r = lo; r < hi; ++r)
            for (int c = lo; c < hi; ++c)
                ++count;
        const real per = (count > 0) ? geo.q_gen_total / real(count) : real(0);
        for (int r = lo; r < hi; ++r)
            for (int c = lo; c < hi; ++c)
                if (r >= 0 && r < n && c >= 0 && c < n)
                    q_gen[idx(5, r, c, n)] = per;

        // ---- cross-panel edge coupling --------------------------------------
        // A cube has 12 edges; each joins two panels along a shared line of n
        // nodes. For the demo we stitch a representative set: join the right
        // column of panel p to the left column of panel p+1, cyclically for the
        // 4 "side" panels (0,1,2,3), forming a ring. Panels 4 and 5 (top/bottom)
        // are joined to panel 0's top/bottom rows. This is a simplified but
        // connected topology — enough to make conduction cross panels.
        for (int p = 0; p < 4; ++p) {
            int q = (p + 1) % 4;
            for (int r = 0; r < n; ++r)
                edges.push_back({ idx(p, r, n - 1, n), idx(q, r, 0, n) });
        }
        for (int c = 0; c < n; ++c) {
            edges.push_back({ idx(0, 0,     c, n), idx(4, n - 1, c, n) }); // top ring->panel4
            edges.push_back({ idx(0, n - 1, c, n), idx(5, 0,     c, n) }); // bottom ring->panel5
        }
    }

    // Initial temperature field: start everything at a uniform 250 K (a plausible
    // cold-ish start) so the transient toward steady state is visible.
    void init_field(std::vector<real>& T) const {
        T.assign(N, real(250.0));
    }
};
