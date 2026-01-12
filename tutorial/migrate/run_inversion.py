#!/usr/bin/env python
"""Run a BayHunter inversion using SurfDisp96 CSV inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_utils import (
    available_columns,
    describe_columns,
    load_dispersion_columns,
    load_velmap,
    save_two_column_file,
)

from BayHunter import PlotFromStorage, Targets, utils
from BayHunter.mcmcOptimizer import MCMC_Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial inversion for Surfdisp96-Tempest-10m CSV files."
    )
    parser.add_argument("--csv", required=True, help="Path to the source CSV file.")
    parser.add_argument(
        "--x-column",
        help="Dispersion x-axis column name (e.g., L_disp_x, M_disp_x).",
    )
    parser.add_argument(
        "--y-column",
        help="Dispersion y-axis column name (e.g., L_disp_y, M_disp_y).",
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config_template.ini"),
        help="Config template with priors and initial parameters.",
    )
    parser.add_argument(
        "--station",
        help="Override station name stored in the config template.",
    )
    parser.add_argument(
        "--savepath",
        help="Override root folder for BayHunter outputs.",
    )
    parser.add_argument("--nchains", type=int, help="Number of MCMC chains.")
    parser.add_argument(
        "--iter-burnin",
        dest="iter_burnin",
        type=int,
        help="Number of burn-in iterations.",
    )
    parser.add_argument(
        "--iter-main",
        dest="iter_main",
        type=int,
        help="Number of main-phase iterations.",
    )
    parser.add_argument(
        "--maxmodels",
        type=int,
        help="Override maximum number of stored posterior models.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=4,
        help="How many chains to evaluate in parallel.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument(
        "--baywatch",
        action="store_true",
        help="Enable BayWatch live streaming.",
    )
    parser.add_argument(
        "--dtsend",
        type=float,
        default=1.0,
        help="How often BayWatch receives updates (seconds).",
    )
    parser.add_argument(
        "--export-dispersion",
        help="Optional two-column text file storing the dispersion curve used for the inversion.",
    )
    parser.add_argument(
        "--velmap-out",
        help="Optional path to store velmap_vs/velmap_z columns for later inspection.",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="List available columns and exit.",
    )
    return parser.parse_args()


def update_initparams(initparams: dict, args: argparse.Namespace) -> None:
    mapping: Tuple[Tuple[str, str], ...] = (
        ("nchains", "nchains"),
        ("iter_burnin", "iter_burnin"),
        ("iter_main", "iter_main"),
        ("station", "station"),
        ("savepath", "savepath"),
        ("maxmodels", "maxmodels"),
    )
    for key, arg_name in mapping:
        value = getattr(args, arg_name, None)
        if value is not None:
            initparams[key] = value


def run_plots(savepath: Path, station: str, maxmodels: int) -> None:
    data_dir = savepath / "data"
    configfile = data_dir / f"{station}_config.pkl"
    if not configfile.exists():
        raise FileNotFoundError(f"BayHunter config file not found: {configfile}")
    plotter = PlotFromStorage(str(configfile))
    plotter.save_final_distribution(maxmodels=maxmodels, dev=0.05)
    plotter.save_plots()
    plotter.merge_pdfs()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if args.list_columns:
        columns = available_columns(csv_path)
        print("Available columns:")
        print(describe_columns(columns))
        return
    if not args.x_column or not args.y_column:
        raise SystemExit("Please provide --x-column and --y-column (use --list-columns to inspect headers).")
    priors, initparams = utils.load_params(args.config)
    update_initparams(initparams, args)
    station = initparams["station"]
    savepath = Path(initparams["savepath"]).resolve()
    savepath.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)
    x, y = load_dispersion_columns(csv_path, args.x_column, args.y_column)
    if args.export_dispersion:
        save_two_column_file(args.export_dispersion, x, y, header=f"{args.x_column} {args.y_column}")
    if args.velmap_out:
        depth, velocity = load_velmap(csv_path)
        save_two_column_file(args.velmap_out, depth, velocity, header="velmap_z velmap_vs")
    target = Targets.RayleighDispersionPhase(x, y)
    targets = Targets.JointTarget(targets=[target])
    print(f"Running inversion for {station}...")
    print(f"Saving results to {savepath}...")
    print(f"Priors: {priors}")
    print(f"Initial parameters: {initparams}")
    print(f"Targets: {target}")
    utils.save_baywatch_config(
        targets,
        path=str(savepath),
        priors=priors,
        initparams=initparams,
    )
    optimizer = MCMC_Optimizer(
        targets,
        initparams=initparams,
        priors=priors,
        random_seed=args.seed,
    )
    print(f"Optimizer: {optimizer}")
    exit()
    optimizer.mp_inversion(
        nthreads=args.nthreads,
        baywatch=args.baywatch,
        dtsend=args.dtsend,
    )
    run_plots(savepath, station, initparams.get("maxmodels", 50000))


if __name__ == "__main__":
    main()
