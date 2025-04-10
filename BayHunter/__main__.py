#
# MIGRATE
#

import os
import numpy as np
import click
import math
import random
import json
import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
import configparser
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from BayHunter.data import SurfDispModel
from BayHunter.data import SeismicPrior, sample_model, SeismicParams, SeismicModel, SeismicSample
from BayHunter.data.huggingface import upload_dataset_to_hf, generate_dataset_card


console = Console()


@click.group()
def cli():
    pass
# end cli


def save_dataset_info(
        output_dir: str,
        dataset_name: str,
        dataset_description: str,
        prior: SeismicPrior,
        params: SeismicParams,
        dispersion_length: int,
        n_samples: int,
        samples_per_shard: int,
        seed: int,
        ini_file: str,
        folds_file: str = "folds.json"
):
    """
    Save dataset metadata information to dataset_info.json.

    :param output_dir: Path where to save the file.
    :param dataset_name: Name of the dataset.
    :param dataset_description: Description of the dataset.
    :param prior: SeismicPrior object used for generation.
    :param params: SeismicParams object used for generation.
    :param dispersion_length: Number of periods in the dispersion curve.
    :param n_samples: Total number of samples generated.
    :param samples_per_shard: Number of samples per shard file.
    :param seed: Random seed used.
    :param ini_file: INI config file path used to generate the dataset.
    :param folds_file: Path to the folds.json file (relative to output_dir).
    """
    n_shards = int(n_samples // samples_per_shard)
    dataset_info = {
        "dataset_name": dataset_name,
        "description": dataset_description,
        "generation": {
            "seed": seed,
            "random_generator": "numpy.default_rng",
            "n_samples": n_samples,
            "samples_per_shard": samples_per_shard,
            "n_shards": n_shards,
            "source": "sample_model + forward",
            "ini_file": ini_file
        },
        "priors": {
            "vs": prior.vs,
            "z": prior.z,
            "layers": prior.layers,
            "vpvs": prior.vpvs,
            "mohoest": prior.mohoest,
            "mantle": prior.mantle,
            "noise_corr": prior.noise_corr,
            "noise_sigma": prior.noise_sigma
        },
        "params": {
            "nchains": params.nchains,
            "iter_burnin": params.iter_burnin,
            "iter_main": params.iter_main,
            "propdist": params.propdist,
            "acceptance": params.acceptance,
            "thickmin": params.thickmin,
            "lvz": params.lvz,
            "hvz": params.hvz,
            "rcond": params.rcond,
            "station": params.station,
            "savepath": params.savepath,
            "maxmodels": params.maxmodels
        },
        "model_parameters": {
            "dispersion_curve_length": dispersion_length
        },
        "features": {
            "vs": {"type": "list<float32>", "variable_length": True},
            "z": {"type": "list<float32>", "variable_length": True},
            "vpvs": {"type": "float32"},
            "disp_x": {"type": "list<float32>", "length": dispersion_length},
            "disp_y": {"type": "list<float32>", "length": dispersion_length},
            "wave_type": {"type": "string"},
            "velocity_type": {"type": "string"}
        },
        "folds": {
            "available": ["2fold", "5fold", "10fold"],
            "fold_file": folds_file
        },
        "format": "parquet",
        "license": "Copyright DMML, HEG Genève 2025",
        "created_by": "Nils Schaetti",
        "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to disk
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)
    # end with

    print("✅ dataset_info.json saved.")
# end save_dataset_info


# Generate folds.json
def generate_folds_json(
        shard_dir: str,
        output_path: str,
        kfolds=None,
        train_ratio: float = 0.8,
        seed: int = 42
):
    """
    Generate folds.json file mapping shards to folds for k-fold cross-validation and train/test split.

    :param shard_dir: Directory containing .parquet shards
    :param output_path: Path to save the folds.json
    :param kfolds: List of k values for k-fold CV
    :param train_ratio: Train/test split ratio (for 2-fold)
    :param seed: Random seed for reproducibility
    """
    # Set seed
    if kfolds is None:
        kfolds = [2, 5, 10]
    # end if

    # Set random seed
    random.seed(seed)

    # Get list of all .parquet shards
    shard_paths = sorted([
        f.name for f in Path(shard_dir).glob("*.parquet")
    ])
    n_shards = len(shard_paths)
    if n_shards == 0:
        raise ValueError("No .parquet shards found in the directory.")
    # end if

    # Shuffle deterministically
    random.shuffle(shard_paths)

    # Folds
    folds = {}

    # 2-fold (train/test)
    split_index = int(train_ratio * n_shards)
    folds["2-fold"] = {
        "train": shard_paths[:split_index],
        "test": shard_paths[split_index:]
    }

    # k-folds
    for k in kfolds:
        if k == 2:
            continue  # Already handled
        # end if
        folds[f"{k}-fold"] = {}
        fold_size = n_shards // k
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n_shards
            folds[f"{k}-fold"][f"fold-{i}"] = shard_paths[start:end]
        # end for
    # end for

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(folds, f, indent=2)
    # end with

    return folds
# end generate_folds_json


@cli.command("generate-dataset")
@click.option("--name", type=str, required=True, help="Name of the dataset")
@click.option("--pretty-name", type=str, required=True, help="Pretty name of the dataset")
@click.option("--description", type=str, required=True, help="Description of the dataset")
@click.option("--license-name", type=str, default="other", help="License name")
@click.option("--repo-id", type=str, default=None, help="Repository ID on Hugging Face Hub")
@click.option("--create-repo/--no-create-repo", default=True, help="Create repo on Hugging Face Hub")
@click.option("--ini-file", type=click.Path(exists=True), required=True, help="Path to .ini file with modelpriors and initparams")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory to save generated dataset")
@click.option("--n-samples", type=int, required=True, help="Total number of samples to generate")
@click.option("--samples-per-shard", type=int, default=10000, help="Number of samples per shard file")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
@click.option("--test-ratio", type=float, default=0.2, help="Test set ratio for 2-fold cross-validation")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
def generate_dataset_cli(
        name,
        pretty_name,
        description,
        license_name,
        repo_id,
        create_repo,
        ini_file,
        output_dir,
        n_samples,
        samples_per_shard,
        length,
        test_ratio,
        seed
):
    """
    Generate synthetic seismic dataset (models + dispersion curves) and save in Arrow format.
    :param name: Name of the dataset
    :param pretty_name: Pretty name of the dataset
    :param description: Description of the dataset
    :param license_name: License name
    :param repo_id: Repository ID on Hugging Face Hub
    :param create_repo: Create repo on Hugging Face Hub
    :param ini_file: Path to .ini file with modelpriors and initparams
    :param output_dir: Directory to save generated dataset
    :param n_samples: Total number of samples to generate
    :param samples_per_shard: Number of samples per shard file
    :param length: Length of the dispersion curve
    :param test_ratio: Test set ratio for 2-fold cross-validation
    :param seed: Random seed for reproducibility
    """
    # Seed
    rng = np.random.default_rng(seed)

    # Load config
    config = configparser.ConfigParser()
    config.read(ini_file)
    prior = SeismicPrior.from_dict(dict(config['modelpriors']))
    params = SeismicParams.from_dict(dict(config['initparams']))

    # Validate parameters
    os.makedirs(output_dir, exist_ok=True)

    # Number of shards
    nshards = math.ceil(n_samples / samples_per_shard)
    click.echo(f"Generating {n_samples} samples in {nshards} shards of {samples_per_shard} each...")

    # Generate samples
    for shard_id in range(nshards):
        samples = []
        for _ in range(samples_per_shard):
            model = sample_model(prior, params, random_seed=rng.integers(0, 1e9))
            curve = model.forward(length=length)
            sample = SeismicSample(model, curve)
            samples.append(sample.to_arrow_dict())
        # end for

        # Convert to Arrow table
        table = pa.table({k: [s[k] for s in samples] for k in samples[0].keys()})
        shard_path = os.path.join(output_dir, f"shard_{shard_id:05d}.parquet")
        pq.write_table(table, shard_path)
        console.print(f"[Shard {shard_id}] Saved {shard_path}")
    # end for

    # Generate folds.json
    generate_folds_json(
        shard_dir=output_dir,
        output_path=os.path.join(output_dir, "folds.json"),
        kfolds=[2, 5, 10],
        train_ratio=1 - test_ratio,
    )

    # Save dataset info
    save_dataset_info(
        output_dir=output_dir,
        dataset_name=name,
        dataset_description=description,
        prior=prior,
        params=params,
        dispersion_length=length,
        n_samples=n_samples,
        samples_per_shard=samples_per_shard,
        seed=seed,
        ini_file=os.path.basename(ini_file)  # relative path
    )

    if repo_id:
        # Determine size category from HuggingFace convention
        if n_samples < 1_000:
            size_category = "n<1K"
        elif n_samples < 10_000:
            size_category = "1K<n<10K"
        elif n_samples < 100_000:
            size_category = "10K<n<100K"
        elif n_samples < 1_000_000:
            size_category = "100K<n<1M"
        elif n_samples < 10_000_000:
            size_category = "1M<n<10M"
        else:
            size_category = "n>10M"
        # end if

        # Create dataset card
        generate_dataset_card(
            output_dir=output_dir,
            license_name=license_name,
            pretty_name=pretty_name,
            size_category=size_category,
            generation_commands=[
                f"python3 -m BayHunter generate-dataset --ini-file {ini_file} --output-dir {output_dir} "
                f"--n-samples {n_samples} --samples-per-shard {samples_per_shard} --seed {seed} "
                f"--repo-id {repo_id}"
            ],
            download_example=True,
            repo_id=repo_id
        )

        # Upload to Hugging Face
        upload_dataset_to_hf(
            output_dir=output_dir,
            repo_id=repo_id,
            private=True,
            commit_message="First upload of synthetic seismic dataset",
            create_repo=create_repo
        )
    # end if

    console.print("\n✅ Dataset generation complete.")
# end generate_dataset


@cli.command("forward-modeling")
@click.option("--model-file", type=click.Path(exists=True), help="Path to a saved seismic model file (.json, .yaml, .pkl)")
@click.option("--ini-file", type=click.Path(exists=True), help="Path to .ini file containing modelpriors and params")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--plot-dispcurve/--no-plot-dispcurve", default=False, help="Plot the dispersion curve")
@click.option("--plot-model/--no-plot-model", default=False, help="Plot the seismic model")
@click.option("--output", type=click.Path(), help="Optional path to save the dispersion curve")
def forward_modeling_cli(model_file, ini_file, length, seed, plot_dispcurve, plot_model, output):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    :param model_file: Path to a saved seismic model file (.json, .yaml, .pkl)
    :param ini_file: Path to .ini file containing modelpriors and params
    :param length: Length of the dispersion curve
    :param seed: Random seed for reproducibility
    :param plot-dispcurve: Whether to plot the dispersion curve
    :param plot-model: Whether to plot the seismic model
    :param output: Optional path to save the dispersion curve
    """
    # Set random seed
    np.random.seed(seed)

    # Load model
    if model_file:
        ext = os.path.splitext(model_file)[1].lower()
        if ext == ".json":
            model = SeismicModel.load_json(model_file)
        elif ext in [".yaml", ".yml"]:
            model = SeismicModel.load_yaml(model_file)
        elif ext in [".pkl", ".pickle"]:
            model = SeismicModel.load_pickle(model_file)
        else:
            raise click.UsageError(f"Unsupported model file extension: {ext}")
        # end if
        console.print(f"[green]Loaded model from {model_file}[/green]")
    elif ini_file:
        config = configparser.ConfigParser()
        config.read(ini_file)

        if 'modelpriors' not in config or 'initparams' not in config:
            raise click.UsageError("Missing [modelpriors] or [initparams] section in ini file.")
        # end if

        prior = SeismicPrior.from_dict(dict(config['modelpriors']))
        params = SeismicParams.from_dict(dict(config['initparams']))
        model = sample_model(prior, params, random_seed=seed)
        console.print("[yellow]No model file provided — model sampled from priors.[/yellow]")
    else:
        raise click.UsageError("You must specify either --model-file or --ini-file.")
    # end if

    # Show the model
    console.print("Seismic Model:")
    console.print(model)

    # Forward modeling
    dc = model.forward(length=length)

    # Output
    console.print("Dispersion Curve:")
    console.print(dc)

    # Plot
    if plot_dispcurve or plot_model:
        if plot_dispcurve and plot_model:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
            model.plot(ax=ax1, invert_axes=False, title="Seismic Model")
            dc.plot(ax=ax2, label="Dispersion", linestyle='-', color='black')
            ax2.legend()
        elif plot_model:
            model.plot(title="Seismic Model")
        elif plot_dispcurve:
            dc.plot()
        # end if
        plt.tight_layout()
        plt.show()
    # end if

    if output:
        ext = os.path.splitext(output)[1].lower()
        if ext == ".json":
            dc.save_json(output)
        elif ext in [".yaml", ".yml"]:
            dc.save_yaml(output)
        elif ext in [".pkl", ".pickle"]:
            dc.save_pickle(output)
        else:
            raise click.UsageError(f"Unsupported output file extension: {ext}")
        # end if
        console.print(f"[green]Dispersion curve saved to {output}[/green]")
    # end if
# end forward_modeling_cli


@cli.command("sample-model")
@click.option("--ini-file", type=click.Path(exists=True), required=True, help="Path to .ini file containing modelpriors")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--output", type=click.Path(), default=None, help="Path to save the model (extension: .json, .yaml, .pkl)")
@click.option("--plot/--no-plot", default=False, help="Plot the sampled model")
def sample_model_cli(ini_file, seed, output, plot):
    """
    Sample a model from a prior specified in an .ini file and print it.

    :param ini_file: Path to the .ini file containing modelpriors
    :param seed: Random seed for reproducibility
    :param output: Path to save the model (extension: .json, .yaml, .pkl)
    :param plot: Whether to display the model plot
    """
    # Set seed
    np.random.seed(seed)

    # Load config
    config = configparser.ConfigParser()
    config.read(ini_file)

    if 'modelpriors' not in config:
        raise click.UsageError("Missing [modelpriors] section in ini file.")
    if 'initparams' not in config:
        raise click.UsageError("Missing [initparams] section in ini file.")
    # end if

    # Load modelpriors and initparams sections
    prior_dict = dict(config['modelpriors'])
    initparams_dict = dict(config['initparams'])

    # Convert to classes
    prior = SeismicPrior.from_dict(prior_dict)
    params = SeismicParams.from_dict(initparams_dict)

    # Sample a model
    model = sample_model(
        prior=prior,
        params=params,
        random_seed=seed
    )

    # Show the sampled model
    console.print("Sampled Seismic Model:")
    console.print(model)

    # Save if needed
    if output:
        ext = os.path.splitext(output)[1].lower()
        if ext == ".json":
            model.save_json(output)
        elif ext in [".yaml", ".yml"]:
            model.save_yaml(output)
        elif ext in [".pkl", ".pickle"]:
            model.save_pickle(output)
        else:
            raise click.UsageError(f"Unsupported file extension: {ext}. Use .json, .yaml, or .pkl")
        # end if
        console.print(f"[green]Model saved to {output}[/green]")
    # end if

    # Plot if requested
    if plot:
        model.plot(title="Sampled Seismic Model", invert_axes=True)
        plt.show()
    # end if
# end sample_model_cli


@cli.command()
@click.option('--h', type=(float, float), required=True, help='Layer thicknesses (km)')
@click.option('--vp', type=(float, float), required=True, help='P-wave velocities (km/s)')
@click.option('--vs', type=(float, float), required=True, help='S-wave velocities (km/s)')
@click.option('--rho', type=(float, float), required=True, help='Layer densities (g/cm³)')
@click.option('--period-min', type=float, default=1.0, help='Minimum period (s)')
@click.option('--period-max', type=float, default=15, help='Maximum period (s)')
@click.option('--period-length', type=int, default=60, help='Number of periods for dispersion calculation')
@click.option('--reference', type=str, default='rdispgr', help='Reference model for dispersion calculation')
def surfdisp96(
        h,
        vp,
        vs,
        rho,
        period_min,
        period_max,
        period_length: int = 60
):
    """
    Run the surfdisp96 model.

    :param period_length:
    :param h: Layer thicknesses (km)
    :param vp: P-wave velocities (km/s)
    :param vs: S-wave velocities (km/s)
    :param rho: Layer densities (g/cm³)
    :param period_min: Minimum period (s)
    :param period_max: Maximum period (s)
    :return: Dispersion velocities (km/s)
    """
    # Run the model
    model = SurfDispModel(
        kmax=period_length,
        min_p=period_min,
        max_p=period_max,
    )

    # Run the model
    x, y = model.run(
        h=np.array(h),
        vp=np.array(vp),
        vs=np.array(vs),
        rho=np.array(rho)
    )

    # Print the results
    table = Table(title="Dispersion Velocities")
    table.add_column("Period (s)", style="bold cyan")
    table.add_column("Dispersion Velocity (km/s)", style="white")
    for period, velocity in zip(x.tolist(), y.tolist()):
        table.add_row(f"{period:.2f}", f"{velocity:.2f}")
    # end for
    console.print(table)
# end surfdisp96


@cli.command()
@click.option('--output', '-o', type=str, required=True, help='Output directory for the generated dataset')
@click.option('--n-samples', '-n', type=int, default=1000, help='Number of samples to generate')
@click.option('--vpvs', type=(float, float), default=(1.4, 2.1), help='Range for Vp/Vs')
@click.option('--vs', type=(float, float), default=(2.0, 5.0), help='Range for Vs values')
@click.option('--z', type=(float, float), default=(0.0, 60.0), help='Range for depth')
@click.option('--layers', type=(int, int), default=(3, 20), help='Range for number of layers')
def generate(
        output,
        n_samples,
        vpvs,

        vs,
        z,
        layers
):
    """
    Generate synthetic models for inverse problem.
    """
    os.makedirs(output, exist_ok=True)

    # Show configuration
    table = Table(title="BayHunter Dataset Configuration")
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Output Dir", output)
    table.add_row("Samples", str(n_samples))
    table.add_row("Vp/Vs", str(vpvs))
    table.add_row("Vs Range", str(vs))
    table.add_row("Depth Range", str(z))
    table.add_row("Number of Layers", str(layers))
    console.print(table)

    for i in range(n_samples):
        # Generate random parameters
        n = np.random.randint(layers[0], layers[1] + 1)
        vs_vals = np.sort(np.random.uniform(vs[0], vs[1], n))
        z_vals = np.sort(np.random.uniform(z[0], z[1], n))
        vpvs_val = np.random.uniform(vpvs[0], vpvs[1])

        # Save the generated sample
        sample = {
            'vs': vs_vals,
            'z': z_vals,
            'vpvs': vpvs_val
        }

        # np.save(os.path.join(output, f'sample_{i:04d}.npy'), sample)
    # end for

    console.print(f"[green]✅ {n_samples} samples generated and saved in '{output}'[/green]")
# end generate


if __name__ == '__main__':
    cli()
# end if
