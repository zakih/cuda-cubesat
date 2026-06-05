#!/usr/bin/env bash
# Build, run the node-count sweep, solve a steady-state field, and render both.
set -e
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build -j
mkdir -p output
./build/cubesat_thermal sweep output/sweep.csv
./build/cubesat_thermal field 128 output/field.bin
python3 scripts/render.py --sweep output/sweep.csv --field output/field.bin
echo "Done. Open output/sweep.html and output/field.html"
