#!/usr/bin/env python
"""Run a BayHunter inversion from SurfDisp96-style CSV inputs.

This tutorial script loads dispersion columns, prepares BayHunter targets,
and runs the MCMC inversion with optional BayWatch streaming.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# end if

from data_utils import (
    available_columns,
    describe_columns,
    load_dispersion_columns,
    load_velmap,
    save_two_column_file,
)

from BayHunter import PlotFromStorage, Targets, utils, PlotFromChains
from BayHunter.mcmcOptimizer import MCMC_Optimizer
from BayHunter.surf96_modsw import SurfDisp


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tutorial inversion."""
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
# end def parse_args


def update_initparams(initparams: dict, args: argparse.Namespace) -> None:
    """Override init params with any CLI-provided values."""
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
        # end if
    # end for
# end def update_initparams


def run_plots_storage(
        savepath: Path,
        station: str,
        maxmodels: int
) -> None:
    """
    Generate BayHunter summary plots from the saved inversion output.
    """
    data_dir = savepath / "data"
    configfile = data_dir / f"{station}_config.pkl"
    if not configfile.exists():
        raise FileNotFoundError(f"BayHunter config file not found: {configfile}")
    # end if
    plotter = PlotFromStorage(configfile=str(configfile))
    plotter.save_final_distribution(maxmodels=maxmodels, dev=0.05)
    plotter.save_plots()
    plotter.merge_pdfs()
# end def run_plots


def run_plots(
        initparams: dict,
        optimizer: MCMC_Optimizer,
        maxmodels: int
) -> None:
    """Generate BayHunter summary plots from the saved inversion output."""
    plotter = PlotFromChains(initparams=initparams, optimizer=optimizer)
    plotter.save_final_distribution(maxmodels=maxmodels, dev=0.05)
    plotter.save_plots()
    plotter.merge_pdfs()
# end def run_plots


def build_stats_table(x: np.ndarray, y: np.ndarray) -> Table:
    """Create a rich table summarizing dispersion curve statistics."""
    table = Table(title="Dispersion Curve Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Points", f"{x.size}")
    table.add_row("x min / max", f"{x.min():.4g} / {x.max():.4g}")
    table.add_row("x mean / std", f"{x.mean():.4g} / {x.std():.4g}")
    table.add_row("y min / max", f"{y.min():.4g} / {y.max():.4g}")
    table.add_row("y mean / std", f"{y.mean():.4g} / {y.std():.4g}")
    return table
# end def build_stats_table


def build_params_table(title: str, params: dict) -> Table:
    """Create a rich table for key/value parameter dictionaries."""
    table = Table(title=title)
    table.add_column("Parameter", style="bold")
    table.add_column("Value")
    for key in sorted(params):
        table.add_row(str(key), repr(params[key]))
    # end for
    return table
# end def build_params_table


def main() -> None:
    """Drive the CSV-to-inversion workflow for the tutorial."""
    args = parse_args()
    csv_path = Path(args.csv)

    if args.list_columns:
        columns = available_columns(csv_path)
        print("Available columns:")
        print(describe_columns(columns))
        return
    # end if

    if not args.x_column or not args.y_column:
        raise SystemExit("Please provide --x-column and --y-column (use --list-columns to inspect headers).")
    # end if

    priors, initparams = utils.load_params(args.config)

    # Apply CLI overrides before preparing folders.
    update_initparams(initparams, args)

    station = initparams["station"]
    savepath = Path(initparams["savepath"]).resolve()
    savepath.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        np.random.seed(args.seed)
    # end if

    # Load dispersion curve columns from the CSV inputs.
    x, y = load_dispersion_columns(csv_path, args.x_column, args.y_column)
    if args.export_dispersion:
        save_two_column_file(args.export_dispersion, x, y, header=f"{args.x_column} {args.y_column}")
    # end if
    if args.velmap_out:
        depth, velocity = load_velmap(csv_path)
        save_two_column_file(args.velmap_out, depth, velocity, header="velmap_z velmap_vs")
    # end if

    console = Console()
    console.print(build_params_table("Init Params", initparams))
    console.print(build_params_table("Priors", priors))
    console.print(build_stats_table(x, y))
    SurfDisp.reset_run_counter()
    if not Confirm.ask("Launch the BayHunter inversion now?", default=True):
        console.print("Inversion canceled by user.")
        return
    # end if

    # Build BayHunter targets from the dispersion curve.
    target = Targets.RayleighDispersionPhase(x, y)
    targets = Targets.JointTarget(targets=[target])

    # Persist BayWatch configuration and run the inversion.
    utils.save_baywatch_config(
        targets,
        path=str(savepath),
        priors=priors,
        initparams=initparams,
    )

    # Create MCMC optimizer
    optimizer = MCMC_Optimizer(
        targets=targets,
        initparams=initparams,
        priors=priors,
        random_seed=args.seed,
    )

    # Launch parallel inversion
    optimizer.mp_inversion(
        nthreads=args.nthreads,
        baywatch=args.baywatch,
        dtsend=args.dtsend,
    )

    # Print final simulations
    for c in optimizer.chains:
        console.print(f"Simulation(s) done: {c.n_simulations}")
        console.print(f"Misfits p1: {c.p1misfits}")
        console.print(f"Misfits p2: {c.p2misfits}")
    # end for

    # Create plots
    run_plots(
        initparams=initparams,
        optimizer=optimizer,
        maxmodels=initparams.get("maxmodels")
    )
# end def main


if __name__ == "__main__":
    main()
# end if
