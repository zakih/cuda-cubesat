#!/usr/bin/env python3
"""Render simulator outputs as interactive Plotly HTML for the GitHub README.

Produces:
  output/sweep.html   : runtime vs node-count, all backends (log-log), plus a
                        speedup-vs-serial panel. Shows the CUDA crossover.
  output/field.html   : steady-state temperature of the 6 CubeSat panels as a
                        grid of heatmaps.

Usage:
    python scripts/render.py --sweep output/sweep.csv --field output/field.bin
"""
import argparse
import struct
import csv
import os
import numpy as np


def read_field(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"CSAT":
            raise ValueError(f"{path}: bad magic {magic!r}")
        n, P, N = struct.unpack("<iii", f.read(12))
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != N:
        raise ValueError(f"{path}: expected {N} values, got {data.size}")
    # reshape to (P, n, n): panel-major, row-major within panel
    return n, P, data.reshape(P, n, n)


def read_sweep(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "n" and k != "N" else int(float(v))
                         for k, v in row.items()})
    return rows


def render_sweep(rows, out_html):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    N = [r["N"] for r in rows]
    series = {
        "serial":     [r["serial_ms"] for r in rows],
        "openmp":     [r["openmp_ms"] for r in rows],
        "cuda_naive": [r["cuda_naive_ms"] for r in rows],
        "cuda_tiled": [r["cuda_tiled_ms"] for r in rows],
    }
    # treat -1 (unavailable, e.g. no CUDA at build) as missing
    def clean(ys):
        return [y if y is not None and y >= 0 else None for y in ys]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Runtime vs node count (log-log)", "Speedup vs serial"))

    colors = {"serial": "#444", "openmp": "#1f77b4",
              "cuda_naive": "#ff7f0e", "cuda_tiled": "#2ca02c"}
    for name, ys in series.items():
        ys = clean(ys)
        fig.add_trace(go.Scatter(x=N, y=ys, name=name, mode="lines+markers",
                                 line=dict(color=colors[name])), row=1, col=1)

    # speedup panel (serial / variant)
    serial = series["serial"]
    for name in ("openmp", "cuda_naive", "cuda_tiled"):
        ys = series[name]
        sp = [(serial[i] / ys[i]) if (ys[i] is not None and ys[i] > 0) else None
              for i in range(len(ys))]
        fig.add_trace(go.Scatter(x=N, y=sp, name=f"{name} speedup", mode="lines+markers",
                                 line=dict(color=colors[name]), showlegend=False), row=1, col=2)
    # speedup=1 reference line
    fig.add_trace(go.Scatter(x=N, y=[1] * len(N), name="1x (serial)", mode="lines",
                             line=dict(color="#999", dash="dash"), showlegend=False), row=1, col=2)

    fig.update_xaxes(type="log", title_text="nodes N", row=1, col=1)
    fig.update_yaxes(type="log", title_text="runtime (ms)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="nodes N", row=1, col=2)
    fig.update_yaxes(type="log", title_text="speedup (×)", row=1, col=2)
    fig.update_layout(title="CubeSat thermal solver: parallelization scaling",
                      template="plotly_white", height=480, width=1000)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"wrote {out_html}")


def render_field(n, P, panels, out_html):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = ["+X (sunlit)", "+Y (sunlit)", "-X", "-Y", "+Z (top)", "-Z (electronics)"]
    vmin = float(panels.min())
    vmax = float(panels.max())
    cols = 3
    rows = (P + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[names[p] if p < len(names) else f"panel {p}"
                                        for p in range(P)])
    for p in range(P):
        r = p // cols + 1
        c = p % cols + 1
        fig.add_trace(go.Heatmap(z=panels[p], zmin=vmin, zmax=vmax,
                                 colorscale="Inferno",
                                 showscale=(p == 0),
                                 colorbar=dict(title="K")),
                      row=r, col=c)
    fig.update_layout(title=f"6U CubeSat steady-state node temperatures (n={n} per panel)",
                      template="plotly_white", height=300 * rows, width=1000)
    # reverse y so row 0 is at the top, like an image
    fig.update_yaxes(autorange="reversed")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"wrote {out_html}  (temp range {vmin:.1f}–{vmax:.1f} K)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="output/sweep.csv")
    ap.add_argument("--field", default="output/field.bin")
    args = ap.parse_args()

    if os.path.exists(args.sweep):
        render_sweep(read_sweep(args.sweep), os.path.splitext(args.sweep)[0] + ".html")
    else:
        print(f"(no sweep csv at {args.sweep}, skipping)")

    if os.path.exists(args.field):
        n, P, panels = read_field(args.field)
        render_field(n, P, panels, os.path.splitext(args.field)[0] + ".html")
    else:
        print(f"(no field bin at {args.field}, skipping)")


if __name__ == "__main__":
    main()
