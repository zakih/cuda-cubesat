# CubeSat Thermal Simulator — Serial vs OpenMP vs CUDA

A transient thermal simulation of a 6U CubeSat, written to compare how OpenMP and CUDA parallelism accelerate a nodal heat-transfer problem. The choice for solution algorithm depends on the nubmer of nodes in the mesh and overhead costs and to find where each one actually helps. This repo explores this empirically by solving the same using four compute backends: serial, OpenMP, a naive CUDA approach, and a tiled CUDA approach which tries to reduce reloads of thermal state for a node and its neighbors from the previous time step. Sample results shown below. The physcis and loading is simplified, future work will expand the problem complexity. 

Steady state thermal results using CUDA kernel

![Steady state results](output/results_field.png)

Performance scaling: Serial vs OpenMP vs CUDA

![Performance scaling results](output/results_sweep.png)



## 1.0 Toy Model case study: 6U CubeSat

A 6U CubeSat (≈10×20×30 cm) is modeled as 6 panels, each meshed into an *n×n* grid of thermal nodes (total *N* ≈ 6*n²*). Each node carries a heat capacity and exchanges energy with its surroundings. Per node, the heat balance is:

$$
C\frac{dT_i}{dt} =
\underbrace{\sum_{j\in\text{neighbors}} k_{ij}(T_j - T_i)}_{\text{conduction}}
\;-\;
\underbrace{\sigma\varepsilon A\,(T_i^4 - T_\text{space}^4)}_{\text{radiation to deep space}}
\;+\;
\underbrace{Q_i^\text{solar}}_{\text{sunlit faces}}
\;+\;
\underbrace{Q_i^\text{gen}}_{\text{electronics}}
$$

- **Conduction**: a 4-neighbor mesh within each panel, plus conduction across the panel edges where faces meet. Linear in temperature.
- **Radiation**: each surface node radiates to the ~3 K deep-space sink via the Stefan–Boltzmann *T⁴* law. This is the nonlinear term.
- **Solar load**: a fixed flux absorbed on the sunlit faces.
- **Internal generation**: electronics/payload dissipation lumped into a patch of nodes on one panel.

Time integration is explicit forward-Euler. The conduction mesh sets a stability limit on the timestep, so Δt is derived from the mesh spacing and stays stable as the mesh is refined.

### 1.1 Modeling Assumptions

This is a deliberately simplified model (for now) and assumes:

1. **No node-to-node radiation exchange / no view factors.** Each surface node radiates only to deep space with an implicit view factor of 1. Panels do not radiate to each other, and there is no self-viewing geometry. This keeps the radiation term purely local, which is also why it parallelizes cleanly. Accelerating the view factors may be a fruitful pursuit however.
2. **Static load case.** Solar flux and internal generation are constant in time. There is no orbital motion modeled therefore no eclipse cycle, or changing sun angle. This is obviously unrealistic. 
3. **Fixed sun geometry.** Two faces are designated "sunlit" with a constant absorbed flux while the rest receive none.
4. **Simplified panel stitching.** I have not modeled the connectivity between sides in a realistic topology, also joint conductance has been simplified to be the same as intra-panel node values. So this does *not* model realistic heat paths. Do not use it for a real study because my motivation was to see performance scaling of different solvers for this math to a first order. 
5. **Uniform material properties.** A single set of properties (k, ρ, cₚ, ε) similar to aluminum are used for every node. Again, edge joints are not treated differently.
6. **Lumped, uniform internal heat.** Electronics dissipation is spread evenly over a fixed patch of nodes rather than modeled as discrete component or payload pieces.
7. **Single precision (float32)** throughout, chosen for GPU throughput. It might be interesting to see how the performance scaling changes at full precision. 



### 1.2 Parallelism methods to accelerate the solver

The insight to enable parallelism is that at each timestep, every node's update depends only on the *previous* step's values of itself and its neighbors (each node has 4 neighbors). No node's new value depends on another node's new value, so a single timestep is embarrassingly parallel and all *N* node updates can happen at once. The simulation is just many such steps in sequence because the time loop itself is serial in that step *t+1* needs step *t*. Here are the four solution methods:

**Serial (baseline)**: one CPU thread loops over all *N* nodes. Runtime scales linearly with *N*

**OpenMP**: a `#pragma omp parallel for` splits the node updates across CPU cores. Whether this pays off depends on *N*: the per-step cost of waking the thread team has to be amortized over enough work. At small and moderate *N* on this problem, this overhead cancels the gain (see plot), but I wanted to show this as well since OpenMP parallelization is a relatively easy implementation.

**CUDA (naive kernel): this uses one GPU thread per node. The GPU's hundreds of cores process the whole grid in parallel, but there's a fixed per-launch cost and sicne the data must obviously live in GPU memory, so the GPU loses at small *N*. The node count where the CUDA runtime curve drops below the CPU curve is what should be used to decide if the project-at-hand should be CUDA accelerated.

**CUDA (shared-memory kernel)**: the memory inefficiency is that adjacent nodes re-read the same neighbor cells. So the shared-memory kernel has each thread block load its tile plus a one-cell halo into on-chip shared memory once, then reads neighbors from there. This is only possible because each panel is stored as a contiguous *n×n* grid which provides a spatial structure to tile. Meaning this approach would break down if the mesh has irregular connectivity. 



### 1.3 Results

Measured on a an ancient NVIDIA GTX 1050 laptop GPU and i7-7700HQ (quadcore 8 logical threads) :


- OpenMP did not do much bettter than serial most of the cases but at low node counts the overhead is too high to justify OpenMP. `OMP_NUM_THREADS=6` was used, but background tasks might have eaten into this. 
- Both the CUDA kernels start outperforming serial after mesh size grows past a few hundred nodes and eventually the speedup goes as high as 30x! 
- This CUDA speadup plateau is very satisfactory, and although this toy 6U CubeSat model won't ever need this many nodes, a sophisticted satellite with tight thermal operational design domain may have millions of nodes. However, note that those more complex models will also need to fix most of the simplifications I have made too. 
- Both CUDA kernels perform almost the same, which isn't surprising given even this GPU we didnot have a bandwidth or caching limitation. Particularly, the L2 cache already absorbs most of the neighbor re-reads the tiling was meant to minimize the impact of. Essentially the cache avoids the issue of reading from slower memory in the first place even for the naive kernel. In fact the shared-memoery kernel has a small overhead (e.g. cooperative load, and sync threads barrier) so it lags slightly behind the naive kernel. The shared-memory kernel would be faster if the global memory bandwidth was an issue, o the problem depended on more neighbors (e.g. large convolution)




### 1.4 Verification

Physics and backend correctness are checked rather than assumed. Two layers:

**Runtime checks:**
- **Cross-backend agreement** — the sweep compares every backend's result against the serial reference and prints `max_abs_diff` per row (in `run_sweep()`, `src/main.cpp`). It should be ~0 to float tolerance; observed values are ≤ 1.5e-5.
- **Convergence criterion** — the field solve runs until max|ΔT| per step falls below tolerance (`Mode::Converge` in the solvers, `src/solver_*.cpp`), and prints the final residual.



### 1.5 Future Work

In rough order of value-to-effort:

1. **Node-to-node radiation with view factors.** The biggest physical upgrade — compute geometric view factors between surfaces (analytical for the simple box, or Monte Carlo ray tracing for fidelity) and add a radiative exchange term. This couples nodes nonlinearly and is far harder to parallelize, which is itself an interesting study.
2. **Dynamic orbital loading.** Make solar flux a function of time: a sun vector sweeping across faces plus an eclipse window. Turns the steady-state solve into a transient cyclic problem; dump frames over an orbit and animate.
3. **Variable duty-cycle internal heat.** Let electronics power switch on/off on a schedule rather than a constant patch.
4. **Implicit time integration.** Replace forward-Euler with backward-Euler / Crank–Nicolson to lift the timestep stability limit — requires a sparse linear solve per step and a very different parallelization story.
5. **Heterogeneous materials and contact resistance.** Per-region k/ρ/cₚ and finite conductance at panel joints.
6. **Max node limit by GPU architecture** Figure out how many nodes can fit into a given GPU VRAM.
7. **double-precision build option** and a study of its accuracy-vs-speed tradeoff.



## 2.0 Setup & Run


### 2.1 Two Run Modes

- **Sweep (timing):** every backend runs a fixed number of steps at each *N*, so the work is identical and the comparison is fair. Writes `output/sweep.csv`.
- **Field (convergence):** runs to steady state (max|ΔT| per step < tolerance) at one resolution and dumps the temperature field. Writes `output/field.bin`. Uses the fastest available backend; all backends compute identical math, so this is purely a speed choice.



### 2.2 Getting started

#### 2.2.1 Host GPU prerequisites (one-time)

Docker needs the NVIDIA Container Toolkit to pass the GPU into a container. The host NVIDIA driver must already be installed (`nvidia-smi` should work). The repo includes `nvidia_container_setup.sh`, which installs and registers the toolkit:

```bash
./nvidia_container_setup.sh
```

Verify the GPU is visible from a container before continuing:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

#### 2.2.2 Build the image

The image is built with your host UID/GID so files written into the mounted volume (build artifacts, CSVs, HTML) are owned by you, not root:

```bash
docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
```

#### 2.2.3 Start the container and build the binary

```bash
docker compose up -d            # starts the environment (it just stays alive)
docker compose exec sim bash    # drop into a shell, or prefix each command below with `docker compose exec sim`
```

Inside the container:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j
```

You should see `CUDA found: building GPU backends (sm_61)` during configure. The executable lands at `build/cubesat_thermal`.

#### 2.2.4 Run

```bash
./build/cubesat_thermal sweep output/sweep.csv     # all four backends, timing sweep
./build/cubesat_thermal field 128 output/field.bin # steady-state solve at n=128
python3 scripts/render.py                           # -> output/sweep.html, output/field.html
```

The `128` is the per-panel mesh resolution *n* (so *N* = 6·128² = 98,304 nodes). The node-count list for the sweep is hardcoded near the top of `run_sweep()` in `src/main.cpp` — edit it to change which sizes run.

Without a CUDA toolkit the build falls back to CPU-only (serial + OpenMP) and the CUDA columns report as unavailable.



### 2.3 Outputs

- `output/sweep.html` — runtime-vs-*N* (log-log) and speedup-vs-serial, all backends.
- `output/field.html` — steady-state temperature of all 6 panels as heatmaps.

Both are self-contained interactive Plotly HTML (Plotly from CDN), so they embed in a GitHub README or any page.



### 2.4 Repository Structure

```
cubesat-thermal/
├── README.md
├── CMakeLists.txt              # auto-detects CUDA; CPU-only fallback
├── Dockerfile
├── docker-compose.yaml
├── nvidia_container_setup.sh   # one-time host GPU/Docker setup
├── src/
│   ├── config.h                # geometry, material props, stability-derived dt, float32
│   ├── geometry.h              # node layout, loads, cross-panel edge coupling
│   ├── node_update.h           # per-node heat balance (shared by all backends)
│   ├── solver.h
│   ├── solver_serial.cpp       # baseline
│   ├── solver_omp.cpp          # OpenMP over nodes
│   ├── solver_cuda.cu          # naive + shared-memory tiled kernels
│   └── main.cpp                # sweep + field driver
├── tests/                      # physics + equivalence checks
├── scripts/
│   ├── render.py               # Plotly: sweep plot + panel temperature maps
│   └── run_all.sh
└── output/                     # csv, bin, html land here
```