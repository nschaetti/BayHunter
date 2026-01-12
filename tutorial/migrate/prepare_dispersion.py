#!/usr/bin/env python
"""Convert SurfDisp96 CSV files into two-column dispersion data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "observed" / "dispersion_from_csv.dat"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_utils import (
    available_columns,
    describe_columns,
    load_dispersion_columns,
    save_two_column_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dispersion columns from Surfdisp96-Tempest CSV files."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file under data/Surfdisp96-Tempest-10m/test/",
    )
    parser.add_argument(
        "--x-column",
        help="Column name containing the dispersion x-axis (e.g., L_disp_x)",
    )
    parser.add_argument(
        "--y-column",
        help="Column name containing the dispersion y-axis (e.g., L_disp_y)",
    )
    parser.add_argument(
        "--output",
        help="Destination for the two-column text file (default: tutorial/migrate/observed/dispersion_from_csv.dat).",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print available columns from the CSV and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if args.list_columns:
        columns = available_columns(csv_path)
        print("Available columns:")
        print(describe_columns(columns))
        return
    if not args.x_column or not args.y_column:
        raise SystemExit("Both --x-column and --y-column are required unless --list-columns is set.")
    x, y = load_dispersion_columns(csv_path, args.x_column, args.y_column)
    outfile = save_two_column_file(
        args.output or DEFAULT_OUTPUT,
        x,
        y,
        header=f"{args.x_column} {args.y_column}",
    )
    print(f"Wrote {x.size} dispersion points to {outfile}")


if __name__ == "__main__":
    main()
