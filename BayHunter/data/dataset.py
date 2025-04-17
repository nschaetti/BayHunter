#
# MIGRATE Project
#

# Imports
import os
import json
import datetime
import random
from pathlib import Path
from typing import Optional, Tuple

from .model import SeismicPrior, SeismicParams


# Create dataset_info.json
def save_dataset_info(
        output_dir: Path,
        dataset_name: str,
        dataset_description: str,
        prior: SeismicPrior,
        params: SeismicParams,
        dispersion_length: int,
        n_samples: int,
        samples_per_shard: int,
        seed: int,
        ini_file: str,
        created_by: str,
        licence: Optional[str] = "other",
        folds_file: Optional[str] = "folds.json",
):
    """
    Save dataset metadata information to dataset_info.json.

    :param output_dir: Path where to save the file.
    :type output_dir: Path
    :param dataset_name: Name of the dataset.
    :type dataset_name: str
    :param dataset_description: Description of the dataset.
    :type dataset_description: str
    :param prior: SeismicPrior object used for generation.
    :type prior: SeismicPrior
    :param params: SeismicParams object used for generation.
    :type params: SeismicParams
    :param dispersion_length: Number of periods in the dispersion curve.
    :type dispersion_length: int
    :param n_samples: Total number of samples generated.
    :type n_samples: int
    :param samples_per_shard: Number of samples per shard file.
    :type samples_per_shard: int
    :param seed: Random seed used.
    :type seed: int
    :param ini_file: INI config file path used to generate the dataset.
    :type ini_file: str
    :param created_by: Name of the person or organization that created the dataset.
    :type created_by: str
    :param licence: License under which the dataset is released.
    :type licence: str
    :param folds_file: Path to the folds.json file (relative to output_dir).
    :type folds_file: str
    """
    # How many shards files are needed?
    n_shards = int(n_samples // samples_per_shard)

    # Create dataset info dictionary
    dataset_info = dict(
        dataset_name=dataset_name,
        description=dataset_description,
        priors=prior.to_dict(),
        params=params.to_dict(["thickmin", "lvz", "hvz"]),
        model_parameters=dict(dispersion_curve_length=dispersion_length),
        folds={"available": ["2fold", "5fold", "10fold"], "fold_file": folds_file},
        format="parquet",
        license=licence,
        created_by=created_by,
        creation_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # Save how we generated the dataset
    dataset_info["generation"] = {
        "seed": seed,
        "random_generator": "numpy.default_rng",
        "n_samples": n_samples,
        "samples_per_shard": samples_per_shard,
        "n_shards": n_shards,
        "source": "sample_model + forward",
        "ini_file": ini_file
    }

    # Save information about features
    dataset_info["features"] = {
        "vs": {"type": "list<float32>", "variable_length": True},
        "z": {"type": "list<float32>", "variable_length": True},
        "vpvs": {"type": "float32"},
        "disp_x": {"type": "list<float32>", "length": dispersion_length},
        "disp_y": {"type": "list<float32>", "length": dispersion_length},
        "wave_type": {"type": "string"},
        "velocity_type": {"type": "string"}
    }

    # Folds

    # Save to disk
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)
    # end with
# end save_dataset_info


# Generate folds.json
def generate_folds_json(
        shard_dir: Path,
        output_path: Path,
        k_folds: Optional[Tuple] = None,
        train_ratio: Optional[float] = 0.8
):
    """
    Generate folds.json file mapping shards to folds for k-fold cross-validation and train/BayHunter_Test split.

    :param shard_dir: Directory containing .parquet shards
    :type shard_dir: Path
    :param output_path: Path to save the folds.json
    :type output_path: Path
    :param k_folds: List of k values for k-fold CV
    :type k_folds: tuple
    :param train_ratio: Train/BayHunter_Test split ratio (for 2-fold)
    :type train_ratio: float
    :param seed: Random seed for reproducibility
    :type seed: int
    :return: Dictionary of folds
    :rtype: dict
    :raise RuntimeError: If no .parquet shards are found in the directory
    """
    # Set seed
    if k_folds is None:
        k_folds = [2, 5, 10]
    # end if

    # Get list of all .parquet shards
    shard_paths = sorted([
        f.name for f in Path(shard_dir).glob("*.parquet")
    ])

    # How many shards do we have?
    n_shards = len(shard_paths)

    # Raise error if no shards found
    if n_shards == 0:
        raise RuntimeError("No .parquet shards found in the directory.")
    # end if

    # Shuffle deterministically
    random.shuffle(shard_paths)

    # Folds
    folds = {}

    # 2-fold (train/BayHunter_Test)
    if 2 in k_folds:
        split_index = int(train_ratio * n_shards)
        folds["2-fold"] = {
            "train": shard_paths[:split_index],
            "test": shard_paths[split_index:]
        }
    # end if

    # k-folds
    for k in k_folds:
        if k == 2:
            continue  # Already handled
        # end if
        folds[f"{k}-fold"] = {}
        fold_size = n_shards // k
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n_shards
            folds[f"{k}-fold"][i] = shard_paths[start:end]
        # end for
    # end for

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(folds, f, indent=2)
    # end with

    return folds
# end generate_folds_json

