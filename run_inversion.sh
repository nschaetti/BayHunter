#!/usr/bin/env bash

set -euo pipefail

for i in $(seq -w 0 99); do
  python3 tutorial/migrate/run_inversion.py \
    --csv data/Surfdisp96-Tempest-10m/test/test0000${i}.csv \
    --x-column H_disp_x \
    --y-column H_disp_y \
    --station tempest_10m_0000${i} \
    --savepath tutorial/migrate/results/tempest_10m_0000${i} \
    --seed 42 \
    --export-dispersion tutorial/migrate/observed/tempest_H_000${i}.dat \
    --velmap-out tutorial/migrate/observed/tempest_H_velmap_0000${i}.dat \
    --config tutorial/migrate/config_migrate.ini \
    --savepath ../../docs/experiments/outputs/MCMC/bayhunter_migrate_test_0000${i}
done
