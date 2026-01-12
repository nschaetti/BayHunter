# Migration Tutorial: SurfDisp96 CSV to BayHunter

This mini tutorial shows how to:

1. Inspect `data/Surfdisp96-Tempest-10m/test/*.csv` files and choose the dispersion columns to invert (e.g., `L_disp_x`, `L_disp_y` for group velocity vs. period);
2. Export those columns as a simple two-column text file if desired;
3. Launch a BayHunter inversion while overriding the key sampler settings (number of chains, iterations, station/save paths);
4. Optionally save the velocity map embedded in the CSV for reference.

All helper scripts live in `tutorial/migrate`.

## 1. Explore the CSV columns

```bash
python tutorial/migrate/run_inversion.py \
  --csv data/Surfdisp96-Tempest-10m/test/test000000.csv \
  --list-columns
```

The command prints every header (e.g., `L_disp_x`, `L_disp_y`, `velmap_vs`, `velmap_z`). Use these names in the next steps.

If you only want to export a dispersion curve for later reuse, run:

```bash
python tutorial/migrate/prepare_dispersion.py \
  --csv data/Surfdisp96-Tempest-10m/test/test000000.csv \
  --x-column L_disp_x \
  --y-column L_disp_y \
  --output tutorial/migrate/observed/tempest_L.dat
```

Omit `--output` to fall back to `tutorial/migrate/observed/dispersion_from_csv.dat`. The resulting two-column file can replace the default BayHunter tutorial data.

## 2. Run the inversion

`tutorial/migrate/run_inversion.py` automates the workflow from raw CSV to posterior plots. Typical usage:

```bash
python tutorial/migrate/run_inversion.py \
  --csv data/Surfdisp96-Tempest-10m/test/test000000.csv \
  --x-column L_disp_x \
  --y-column L_disp_y \
  --station tempest_demo \
  --savepath tutorial/migrate/results/tempest_demo \
  --nchains 6 \
  --iter-burnin $((2048 * 4)) \
  --iter-main $((2048 * 4)) \
  --nthreads 6 \
  --seed 42 \
  --export-dispersion tutorial/migrate/observed/tempest_demo.dat \
  --velmap-out tutorial/migrate/observed/tempest_demo_velmap.dat
```

Key flags:

- `--config tutorial/migrate/config_template.ini`: optional template with priors and defaults (used automatically if omitted).
- `--station` / `--savepath`: label and destination folder for BayHunter outputs.
- `--nchains`, `--iter-burnin`, `--iter-main`, `--maxmodels`: override sampler settings without touching the `.ini`.
- `--baywatch` `--dtsend`: enable live monitoring if BayWatch is running.
- `--export-dispersion`: persist the curve used in the inversion.
- `--velmap-out`: dump `velmap_z` / `velmap_vs` so you can visualize the starting models.

After `mp_inversion` finishes, the script automatically calls `PlotFromStorage` to generate posterior distributions, dispersion-fit plots, and merged PDFs inside `<savepath>/plots`.

## 3. Results

- Inversion logs, misfits, and posterior models are stored under `<savepath>`.
- The BayWatch configuration is saved to `<savepath>/baywatch.pkl`.
- Posterior plots/PDFs live under `<savepath>/plots` and `<savepath>/posterior_distributions`.

Use these scripts as templates to migrate any SurfDisp96-derived CSV into the BayHunter workflow. Adjust the priors in `config_template.ini` if your target site requires different velocity bounds, layer counts, or noise assumptions.
