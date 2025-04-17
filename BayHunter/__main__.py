#
# MIGRATE
#

import os
import numpy as np
import click
import math
import random
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import configparser
from pathlib import Path
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

# Imports BayHunter
from BayHunter.data import SurfDispModel
from BayHunter.data import SeismicPrior, sample_model, SeismicParams, SeismicModel, SeismicSample
from BayHunter.data.huggingface import upload_dataset_to_hf, generate_dataset_card
from BayHunter.data import save_dataset_info, generate_folds_json


console = Console()
install(show_locals=True)


@click.group()
def cli():
    pass
# end cli


# Convert tuple of integers to a list of integers
def tuple_of_ints(value):
    """
    Convert a string representation of a tuple to an actual tuple of integers.

    :param value: String representation of a tuple (e.g., "2,5,10)")
    :return: Tuple of integers
    """
    if isinstance(value, str):
        value = value.split(",")
        value = [int(v) for v in value]
    # end if
    return tuple(value)
# end tuple_of_ints


@cli.command("generate-dataset")
@click.option("--name", type=str, required=True, help="Name of the dataset")
@click.option("--pretty-name", type=str, required=True, help="Pretty name of the dataset")
@click.option("--description", type=str, required=True, help="Description of the dataset")
@click.option("--license-name", type=str, default="other", help="License name")
@click.option("--created-by", type=str, default="Unknown", help="Name of the creator")
@click.option("--ini-file", type=click.Path(exists=True), required=True, help="Path to .ini file with modelpriors and initparams")
@click.option("--output-dir", type=click.Path(), required=True, help="Directory to save generated dataset")
@click.option("--n-samples", type=int, required=True, help="Total number of samples to generate")
@click.option("--samples-per-shard", type=int, default=10000, help="Number of samples per shard file")
@click.option("--length", type=int, default=108, help="Length of the dispersion curve")
@click.option("--test-ratio", type=float, default=0.2, help="Test set ratio for 2-fold cross-validation")
@click.option("--folds", type=tuple_of_ints, default=(2, 5, 10), help="List of k values for k-fold cross-validation (ex: 2,5,10)")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--write-dataset-info", is_eager=True, is_flag=True, help="Write JSON files")
@click.option("--write-folds", is_eager=True, is_flag=True, help="Write folds.json")
@click.option("--write-dataset-card", is_eager=True, is_flag=True, help="Write dataset card")
def generate_dataset_cli(
        name,
        pretty_name,
        description,
        license_name,
        created_by: str,
        ini_file,
        output_dir,
        n_samples,
        samples_per_shard,
        length,
        test_ratio,
        folds,
        seed,
        write_dataset_info: bool,
        write_folds: bool,
        write_dataset_card: bool
):
    """
    Generate synthetic seismic dataset (models + dispersion curves) and save in Arrow format.

    :param name: Name of the dataset
    :type name: str
    :param pretty_name: Pretty name of the dataset
    :type pretty_name: str
    :param description: Description of the dataset
    :type description: str
    :param license_name: License name
    :type license_name: str
    :param created_by: Name of the creator
    :type created_by: str
    :param ini_file: Path to .ini file with modelpriors and initparams
    :type ini_file: str
    :param output_dir: Directory to save generated dataset
    :type output_dir: str
    :param n_samples: Total number of samples to generate
    :type n_samples: int
    :param samples_per_shard: Number of samples per shard file
    :type samples_per_shard: int
    :param length: Length of the dispersion curve
    :type length: int
    :param test_ratio: Test set ratio for 2-fold cross-validation
    :type test_ratio: float
    :param folds: List of k values for k-fold cross-validation (ex: 2,5,10)
    :type folds: tuple
    :param seed: Random seed for reproducibility
    :type seed: int
    :param write_dataset_info: Write JSON files
    :type write_dataset_info: bool
    :param write_folds: Write folds.json
    :type write_folds: bool
    :param write_dataset_card: Write dataset card
    :type write_dataset_card: bool
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # Output directory
    output_dir = Path(output_dir)

    # Load config
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Get prior and params
    prior = SeismicPrior.from_dict(dict(config['modelpriors']))
    params = SeismicParams.from_dict(dict(config['initparams']))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Number of shards
    nshards = math.ceil(n_samples / samples_per_shard)
    console.log(f"Generating {n_samples} samples in {nshards} shards of {samples_per_shard} each...")

    # Generate samples
    for shard_id in range(nshards):
        samples = []
        for _ in range(samples_per_shard):
            model = sample_model(prior, params, random_seed=np.random.randint(0, 2**32 - 1))
            curve = model.forward(length=length)
            sample = SeismicSample(model, curve)
            samples.append(sample.to_arrow_dict())
        # end for

        # Convert to Arrow table
        table = pa.table({k: [s[k] for s in samples] for k in samples[0].keys()})
        shard_path = output_dir / f"shard_{shard_id:05d}.parquet"
        pq.write_table(table, str(shard_path))
        console.log(f"[Shard {shard_id}] Saved {shard_path}")
    # end for

    # Generate folds.json
    if write_folds:
        generate_folds_json(
            shard_dir=output_dir,
            output_path=output_dir / "folds.json",
            k_folds=folds,
            train_ratio=1 - test_ratio,
        )
        console.log(f"Generated folds.json in {output_dir}")
    # end if

    # Save dataset info
    if write_dataset_info:
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
            ini_file=os.path.basename(ini_file),
            created_by=created_by
        )
        console.log(f"Saved dataset_info.json in {output_dir}")
    # end if

    if write_dataset_card:
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
                f"--n-samples {n_samples} --samples-per-shard {samples_per_shard} --seed {seed} --length {length} "
                f"--test-ratio {test_ratio} --folds {','.join(map(str, folds))}"
            ],
            download_example=True
        )
        console.log(f"Generated dataset card in {output_dir}")
    # end if

    # if repo_id:
    #     # Upload to Hugging Face
    #     upload_dataset_to_hf(
    #         output_dir=output_dir,
    #         repo_id=repo_id,
    #         private=True,
    #         commit_message="First upload of synthetic seismic dataset",
    #         create_repo=create_repo
    #     )
    # # end if

    console.log("Dataset generation complete.")
    console.log(
        "You can now upload the dataset on HuggingFace with: "
        "`huggingface-cli upload MIGRATE/<dataset-name> <loca-path> . --repo-type dataset "
        "--commit-message \"<commit-message>\" --private`"
    )
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
