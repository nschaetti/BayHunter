#!/usr/bin/env bash

set -euo pipefail

for i in $(seq -w 0 99); do
  python3 tutorial/migrate/run_inversion.py \
    --csv data/Surfdisp96-Tempest-10m/test/test0000${i}.csv \
    --x-column H_disp_x \
    --y-column H_disp_y \
    --station tempest_10m_0000${i} \
    --seed 42 \
    --export-dispersion tutorial/migrate/observed/tempest_H_000${i}.dat \
    --velmap-out tutorial/migrate/observed/tempest_H_velmap_0000${i}.dat \
    --config tutorial/migrate/config_migrate_sim_data.ini \
    --savepath ../../docs/experiments/outputs/MCMC/sim_data/bayhunter_migrate_sim_data_0000${i}
done
