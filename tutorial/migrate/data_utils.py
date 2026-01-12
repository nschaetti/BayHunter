"""Helpers for turning dispersion CSV files into BayHunter inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def available_columns(csv_path: str | Path) -> List[str]:
    """Return the header columns stored in the CSV file."""
    path = Path(csv_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return [column.strip() for column in header]


def load_dispersion_columns(
    csv_path: str | Path,
    x_column: str,
    y_column: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dispersion data columns as sorted numpy arrays."""
    path = Path(csv_path)
    xs: List[float] = []
    ys: List[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_value = _coerce_float(row.get(x_column))
            y_value = _coerce_float(row.get(y_column))
            if x_value is None or y_value is None:
                continue
            xs.append(x_value)
            ys.append(y_value)
    if not xs:
        raise ValueError(
            f"No numeric data found in columns '{x_column}'/'{y_column}' of {csv_path}"
        )
    order = np.argsort(xs)
    x_sorted = np.asarray(xs)[order]
    y_sorted = np.asarray(ys)[order]
    return x_sorted, y_sorted


def load_velmap(
    csv_path: str | Path,
    depth_column: str = "velmap_z",
    velocity_column: str = "velmap_vs",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return depth-velocity arrays from velmap columns."""
    z, vel = load_dispersion_columns(csv_path, depth_column, velocity_column)
    return z, vel


def save_two_column_file(
    outfile: str | Path,
    x: Sequence[float],
    y: Sequence[float],
    header: str | None = None,
) -> Path:
    """Write two ordered columns to disk."""
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((x, y))
    np.savetxt(path, data, fmt="%.8f", header=header or "", comments="")
    return path


def describe_columns(columns: Iterable[str]) -> str:
    """Return a printable description for logging."""
    return "\n".join(f"- {name}" for name in columns)

