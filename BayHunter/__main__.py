#
# MIGRATE
#

import os
import numpy as np
import click
import math
import concurrent.futures
import random
from matplotlib import cm
from rich.console import Console
from rich.table import Table
from rich.traceback import install
import configparser
from pathlib import Path
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Imports BayHunter
from BayHunter.data import SurfDispModel
from BayHunter.data import SeismicPrior, sample_model, SeismicParams, SeismicModel, SeismicSample
from BayHunter.data import vornoi_to_layers
from BayHunter.data.huggingface import upload_dataset_to_hf, generate_dataset_card
from BayHunter.data import save_dataset_info, generate_folds_json, validate_model


console = Console()
install(show_locals=False)


# Simulation Timeout
SIM_TIMEOUT = 2.0


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


class FloatListParamType(click.ParamType):

    name = "float_list"

    def convert(self, value, param, ctx):
        try:
            return [float(v.strip()) for v in value.split(",")]
        except ValueError:
            self.fail(f"{value} is not a valid comma-separated list of floats", param, ctx)
        # end try
    # end convert

# end FloatListParamType
FLOAT_LIST = FloatListParamType()


def safe_forward(
        model,
        length
):
    """
    Run the forward method of the model in a separate thread to avoid blocking the main thread.

    Args:
        model (SeismicModel): The seismic model to run the forward method on.
        length (int): The length of the dispersion curve.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model.forward, length=length)
        try:
            return future.result(timeout=SIM_TIMEOUT)
        except concurrent.futures.TimeoutError:
            raise TimeoutError("model.forward took too long and was terminated.")
        # end try
    # end with
# end safe_forward


@cli.command("plot-2d-perturbations")
@click.option("--vpvs", type=float, default=1.75, help="Vp/Vs ratio")
@click.option("--vs", type=FLOAT_LIST, required=True, help="List of Vs values in Km/s (e.g., --vs 2.174 2.46)")
@click.option("--z", type=FLOAT_LIST, required=True, help="List of Z values in Km (e.g., --z 1.24 10.91)")
@click.option("--vs-noise", type=float, default=0.1, help="Noise on Vs values")
@click.option("--vs-noise-samples", type=int, default=50, help="Number of samples to generate for Vs noise")
@click.option("--z-noise", type=float, default=0.1, help="Noise on Z values")
@click.option("--z-noise-samples", type=int, default=50, help="Number of samples to generate for Z noise")
@click.option("--n-samples", type=int, default=100, help="Number of samples to generate")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
def plot_2d_perturbations(
        vpvs,
        vs,
        z,
        vs_noise,
        vs_noise_samples,
        z_noise,
        z_noise_samples,
        n_samples,
        length,
        seed: int = 42
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    Args:
        vpvs (float): Vp/Vs ratio
        vs (tuple): Tuple of Vs values (velocity)
        z (tuple): Tuple of depth values (depth)
        vs_noise (float): Noise on Vs values
        vs_noise_samples (int): Number of samples to generate for Vs noise
        z_noise (float): Noise on Vs values
        z_noise_samples (int): Number of samples to generate for Vs noise
        n_samples (int): Number of samples to generate
        length (int): Length of the dispersion curve
        seed (int): Random seed for reproducibility
    """
    # Set seed
    np.random.seed(seed)

    # To numpy array
    vs = np.array(vs)
    z = np.array(z)

    # Check if Vs and Z are the same length
    if len(vs) != len(z):
        raise click.UsageError("The number of Vs and Z values must be the same.")
    # end if

    # Log number of layers
    console.print(f"[yellow]Number of layers: {vs.shape[0]}[/yellow]")

    # List of model and dispersion curves
    misfits = np.zeros((vs_noise_samples, z_noise_samples, n_samples))

    # Base model
    base_model = SeismicModel(
        model=np.concatenate((vs, z)),
        vpvs=vpvs
    )
    base_dc = base_model.forward(length=length)

    # Total iterations for progress bar
    total = vs_noise_samples * z_noise_samples * n_samples
    progress_bar = tqdm(total=total, desc="Generating models", unit="sample")

    # For each noise level
    for j, vs_nl in enumerate(np.linspace(0, vs_noise, vs_noise_samples)):
        for k, z_nl in enumerate(np.linspace(0, z_noise, z_noise_samples)):
            # For each sample
            for i in range(n_samples):
                ok = False
                retry = 0
                while not ok:
                    # Put Vs and layers together
                    model = SeismicModel(
                        model=np.concatenate(
                            (
                                vs + np.random.randn(vs.shape[0]) * vs_nl,
                                z + np.random.randn(z.shape[0]) * z_nl
                            )
                        ),
                        vpvs=vpvs
                    )

                    # Forward modeling
                    try:
                        dc = model.forward(length=length)
                        ok = True
                    except TypeError as e:
                        retry += 1
                        pass
                    except TimeoutError as e:
                        dc = None
                        break
                    # end try

                    if retry > 10:
                        dc = None
                        ok = True
                    # end if
                # end while

                # Compute misfit with the base model
                if dc:
                    misfit = dc.misfit(base_dc)
                else:
                    misfit = np.nan
                # end if

                # Append
                misfits[j, k, i] = misfit
                progress_bar.update(1)
            # end for
        # end for
    # end for

    progress_bar.close()

    # Average misfit per noise level
    avg_misfits = np.mean(misfits, axis=-1)

    # Cut values above 100
    avg_misfits[avg_misfits > 100] = 100

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Plot average misfit per noise level with a hotmap
    ax2.imshow(
        avg_misfits,
        aspect='auto',
        cmap='hot',
        interpolation='nearest',
        extent=[0, z_noise, 0, vs_noise],
        origin='lower',
        vmin=0,
        vmax=100
    )

    # Titles
    ax1.set_title("Seismic Model")
    ax2.set_title(f"Misfit vs Noise Level")

    # Labels
    ax2.set_xlabel("Z Noise Level")
    ax2.set_ylabel("Vs Noise Level")

    # Plot base model
    base_model.plot(ax=ax1, invert_axes=False, title="Seismic Model", linewidth=4, color='black')

    # ax2.legend()
    plt.tight_layout()
    plt.show()
# end plot_2d_perturbations



def run_perturbations(
        vpvs,
        vs,
        z,
        noise,
        noise_samples,
        noise_target: str,
        n_samples,
        length,
        seed: int = 42
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.
    """
    assert noise_target in ["vs", "z"], "noise_target must be either 'vs' or 'z'"

    # Set seed
    np.random.seed(seed)

    # To numpy array
    vs = np.array(vs)
    z = np.array(z)

    # Check if Vs and Z are the same length
    if len(vs) != len(z):
        raise click.UsageError("The number of Vs and Z values must be the same.")
    # end if

    # Log number of layers
    console.print(f"[yellow]Number of layers: {vs.shape[0]}[/yellow]")

    # List of model and dispersion curves
    misfits = np.zeros((noise_samples, n_samples))

    # Base model
    base_model = SeismicModel(
        model=np.concatenate((vs, z)),
        vpvs=vpvs
    )
    base_dc = base_model.forward(length=length)

    # For each noise level
    for j, nl in enumerate(np.linspace(0, noise, noise_samples)):
        print(f"[green]Noise level: {nl}[/green]")
        # For each sample
        for i in range(n_samples):
            ok = False
            retry = 0
            while not ok:
                # Put Vs and layers together
                model = SeismicModel(
                    model=np.concatenate(
                        (
                            vs + np.random.randn(vs.shape[0]) * (nl if noise_target == "vs" else 0.0),
                            z + np.random.randn(z.shape[0]) * (nl if noise_target == "z" else 0.0)
                        )
                    ),
                    vpvs=vpvs
                )

                # Forward modeling
                try:
                    dc = model.forward(length=length)
                    ok = True
                except TypeError as e:
                    retry += 1
                    pass
                # end try

                if retry > 10:
                    console.print(f"[red]Failed to generate model after {retry} retries.[/red]")
                    exit(1)
                # end if
            # end while

            # Compute misfit with the base model
            misfit = dc.misfit(base_dc)

            # Append
            misfits[j, i] = misfit
        # end for
    # end for

    # Average misfit per noise level
    avg_misfits = np.mean(misfits, axis=1)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Plot average misfit per noise level
    ax2.plot(
        np.linspace(0, noise, noise_samples),
        avg_misfits,
        label="Average Misfit",
        linestyle='-',
        color='black',
        linewidth=4
    )

    # Titles
    ax1.set_title("Seismic Model")
    ax2.set_title(f"Misfit vs Noise Level for {noise_target}")

    # Plot base model
    base_model.plot(ax=ax1, invert_axes=False, title="Seismic Model", linewidth=4, color='black')

    # ax2.legend()
    plt.tight_layout()
    plt.show()
# end run_perturbations


@cli.command("plot-z-perturbations")
@click.option("--vpvs", type=float, default=1.75, help="Vp/Vs ratio")
@click.option("--vs", type=FLOAT_LIST, required=True, help="List of Vs values in Km/s (e.g., --vs 2.174 2.46)")
@click.option("--z", type=FLOAT_LIST, required=True, help="List of Z values in Km (e.g., --z 1.24 10.91)")
@click.option("--z-noise", type=float, default=0.1, help="Noise on Vs values")
@click.option("--z-noise-samples", type=int, default=50, help="Number of samples to generate for Vs noise")
@click.option("--n-samples", type=int, default=100, help="Number of samples to generate")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
def plot_vs_perturbations(
        vpvs,
        vs,
        z,
        z_noise,
        z_noise_samples,
        n_samples,
        length,
        seed: int = 42
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    Args:
        vpvs (float): Vp/Vs ratio
        vs (tuple): Tuple of Vs values (velocity)
        z (tuple): Tuple of depth values (depth)
        z_noise (float): Noise on Vs values
        z_noise_samples (int): Number of samples to generate for Vs noise
        n_samples (int): Number of samples to generate
        length (int): Length of the dispersion curve
        seed (int): Random seed for reproducibility
    """
    run_perturbations(
        vpvs=vpvs,
        vs=vs,
        z=z,
        noise=z_noise,
        noise_samples=z_noise_samples,
        noise_target="z",
        n_samples=n_samples,
        length=length,
        seed=seed
    )
# end run_forward_cli


@cli.command("plot-vs-perturbations")
@click.option("--vpvs", type=float, default=1.75, help="Vp/Vs ratio")
@click.option("--vs", type=FLOAT_LIST, required=True, help="List of Vs values in Km/s (e.g., --vs 2.174 2.46)")
@click.option("--z", type=FLOAT_LIST, required=True, help="List of Z values in Km (e.g., --z 1.24 10.91)")
@click.option("--vs-noise", type=float, default=0.1, help="Noise on Vs values")
@click.option("--vs-noise-samples", type=int, default=50, help="Number of samples to generate for Vs noise")
@click.option("--n-samples", type=int, default=100, help="Number of samples to generate")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
def plot_vs_perturbations(
        vpvs,
        vs,
        z,
        vs_noise,
        vs_noise_samples,
        n_samples,
        length,
        seed: int = 42
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    Args:
        vpvs (float): Vp/Vs ratio
        vs (tuple): Tuple of Vs values (velocity)
        z (tuple): Tuple of depth values (depth)
        vs_noise (float): Noise on Vs values
        vs_noise_samples (int): Number of samples to generate for Vs noise
        n_samples (int): Number of samples to generate
        length (int): Length of the dispersion curve
        seed (int): Random seed for reproducibility
    """
    run_perturbations(
        vpvs=vpvs,
        vs=vs,
        z=z,
        noise=vs_noise,
        noise_samples=vs_noise_samples,
        noise_target="vs",
        n_samples=n_samples,
        length=length,
        seed=seed
    )
# end run_forward_cli


@cli.command("noisy-forward")
@click.option("--vpvs", type=float, default=1.75, help="Vp/Vs ratio")
@click.option("--vs", type=FLOAT_LIST, required=True, help="List of Vs values in Km/s (e.g., --vs 2.174 2.46)")
@click.option("--z", type=FLOAT_LIST, required=True, help="List of Z values in Km (e.g., --z 1.24 10.91)")
@click.option("--vs-noise", type=float, default=0.1, help="Noise on Vs values")
@click.option("--z-noise", type=float, default=0.1, help="Noise on Z values")
@click.option("--n-samples", type=int, default=100, help="Number of samples to generate")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
def run_forward_cli(
        vpvs,
        vs,
        z,
        vs_noise,
        z_noise,
        n_samples,
        length,
        seed: int = 42
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    Args:
        vpvs (float): Vp/Vs ratio
        vs (tuple): Tuple of Vs values (velocity)
        z (tuple): Tuple of depth values (depth)
        vs_noise (float): Noise on Vs values
        z_noise (float): Noise on Z values
        n_samples (int): Number of samples to generate
        length (int): Length of the dispersion curve
        seed (int): Random seed for reproducibility
    """
    # Set seed
    np.random.seed(seed)

    # To numpy array
    vs = np.array(vs)
    z = np.array(z)

    # Check if Vs and Z are the same length
    if len(vs) != len(z):
        raise click.UsageError("The number of Vs and Z values must be the same.")
    # end if

    # Log number of layers
    console.print(f"[yellow]Number of layers: {vs.shape[0]}[/yellow]")

    # List of model and dispersion curves
    models = []
    dcs = []
    misfits = np.zeros((n_samples,))

    # Base model
    model = SeismicModel(
        model=np.concatenate((vs, z)),
        vpvs=vpvs
    )
    dc = model.forward(length=length)

    # Add
    models.append(model)
    dcs.append(dc)

    # For each sample
    for i in range(n_samples):
        # Put Vs and layers together
        model = SeismicModel(
            model=np.concatenate(
                (
                    vs + np.random.randn(vs.shape[0]) * vs_noise,
                    z + np.random.randn(z.shape[0]) * z_noise
                )
            ),
            vpvs=vpvs
        )

        # Forward modeling
        dc = model.forward(length=length)

        # Compute misfit with the base model
        misfit = dc.misfit(dcs[0])

        # Append
        models.append(model)
        dcs.append(dc)
        misfits[i] = misfit
    # end for

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

    # Generate a colormap excluding black (we use 'tab10' or 'viridis' for clarity)
    cmap = cm.get_cmap('tab10')  # Or 'tab20', 'Set1', etc.
    num_colors = len(models) - 2
    colors = [cmap(i % cmap.N) for i in range(1, num_colors + 1)]  # Start from index 1 to skip black

    # Plot all models
    for i, (model, color) in enumerate(zip(models[2:], colors), start=1):
        model.plot(ax=ax1, invert_axes=False, title="Seismic Model", linewidth=0.5, color=color, alpha=1)
        dcs[i + 1].plot(ax=ax2, label=f"Dispersion {i + 1}", linestyle='-', color=color, linewidth=0.5, alpha=1)
    # end for

    # Plot base model
    models[0].plot(ax=ax1, invert_axes=False, title="Seismic Model", linewidth=4, color='black')
    dcs[0].plot(ax=ax2, label="Dispersion", linestyle='-', color='black', linewidth=4)

    # Show average misfit in legend
    avg_misfit = np.mean(misfits)
    ax2.legend(title=f"Avg. Misfit: {avg_misfit:.2f}", loc='upper right')

    # ax2.legend()
    plt.tight_layout()
    plt.show()
# end run_forward_cli


@cli.command("run-forward")
@click.option("--vpvs", type=float, default=1.75, help="Vp/Vs ratio")
@click.option("--vs", type=FLOAT_LIST, required=True, help="List of Vs values in Km/s (e.g., --vs 2.174 2.46)")
@click.option("--z", type=FLOAT_LIST, required=True, help="List of Z values in Km (e.g., --z 1.24 10.91)")
@click.option("--length", type=int, default=60, help="Length of the dispersion curve")
@click.option("--plot-dispcurve/--no-plot-dispcurve", default=False, help="Plot the dispersion curve")
@click.option("--plot-model/--no-plot-model", default=False, help="Plot the seismic model")
@click.option("--curve-output", type=click.Path(), help="Optional path to save the dispersion curve")
@click.option("--model-output", type=click.Path(), help="Optional path to save the seismic model")
def run_forward_cli(
        vpvs,
        vs,
        z,
        length,
        plot_dispcurve,
        plot_model,
        curve_output,
        model_output
):
    """
    Run the forward modeling process using either a saved model or priors from an ini file.

    Args:
        vpvs (float): Vp/Vs ratio
        vs (tuple): Tuple of Vs values (velocity)
        z (tuple): Tuple of depth values (depth)
        length (int): Length of the dispersion curve
        plot_dispcurve (bool): Whether to plot the dispersion curve
        plot_model (bool): Whether to plot the seismic model
        curve_output (str): Optional path to save the dispersion curve
        model_output (str): Optional path to save the seismic model
    """
    # To numpy array
    vs = np.array(vs)
    z = np.array(z)

    # Check if Vs and Z are the same length
    if len(vs) != len(z):
        raise click.UsageError("The number of Vs and Z values must be the same.")
    # end if

    # Log number of layers
    console.print(f"[yellow]Number of layers: {vs.shape[0]}[/yellow]")

    # Put Vs and layers together
    model = SeismicModel(
        model=np.concatenate((vs, z)),
        vpvs=vpvs
    )

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

    if curve_output:
        ext = os.path.splitext(curve_output)[1].lower()
        if ext == ".json":
            dc.save_json(curve_output)
        elif ext in [".yaml", ".yml"]:
            dc.save_yaml(curve_output)
        elif ext in [".pkl", ".pickle"]:
            dc.save_pickle(curve_output)
        else:
            raise click.UsageError(f"Unsupported output file extension: {ext}")
        # end if
        console.print(f"[green]Dispersion curve saved to {curve_output}[/green]")
    # end if

    if model_output:
        ext = os.path.splitext(model_output)[1].lower()
        if ext == ".json":
            model.save_json(model_output)
        elif ext in [".yaml", ".yml"]:
            model.save_yaml(model_output)
        elif ext in [".pkl", ".pickle"]:
            model.save_pickle(model_output)
        else:
            raise click.UsageError(f"Unsupported output file extension: {ext}")
        # end if
        console.print(f"[green]Model saved to {model_output}[/green]")
    # end if
# end run_forward_cli



@cli.command("generate-dataset")
@click.option("--name", type=str, required=False, help="Name of the dataset")
@click.option("--pretty-name", type=str, required=False, help="Pretty name of the dataset")
@click.option("--description", type=str, required=False, help="Description of the dataset")
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
@click.option("--variable-grid", is_eager=True, is_flag=True, help="Use variable grid for dispersion curves (default: False)")
@click.option("--min-p", type=float, default=1.0, help="Minimum period for dispersion curves (default: 1.0)")
@click.option("--max-p", type=float, default=15.0, help="Maximum period for dispersion curves (default: 15.0)")
@click.option("--min-period-difference", type=float, default=10.0, help="Minimum period difference for dispersion curves (default: 10.0)")
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
        write_dataset_card: bool,
        variable_grid: bool = False,
        min_p: float = 1.0,
        max_p: float = 15.0,
        min_period_difference: float = 10.0
):
    """
    Generate synthetic seismic dataset (models + dispersion curves) and save in Arrow format.

    Args:
        name (str): Name of the dataset
        pretty_name (str): Pretty name of the dataset
        description (str): Description of the dataset
        license_name (str): License name
        created_by (str): Name of the creator
        ini_file (str): Path to .ini file with modelpriors and initparams
        output_dir (str): Directory to save generated dataset
        n_samples (int): Total number of samples to generate
        samples_per_shard (int): Number of samples per shard file
        length (int): Length of the dispersion curve
        test_ratio (float): Test set ratio for 2-fold cross-validation
        folds (tuple): List of k values for k-fold cross-validation (ex: 2,5,10)
        seed (int): Random seed for reproducibility
        write_dataset_info (bool): Write JSON files
        write_folds (bool): Write folds.json
        write_dataset_card (bool): Write dataset card
        variable_grid (bool): Use variable grid for dispersion curves (default: False)
        min_p (float): Minimum period for dispersion curves (default: 1.0)
        max_p (float): Maximum period for dispersion curves (default: 15.0)
        min_period_difference (float): Minimum period difference for dispersion curves (default: 10.0)
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
        for i in range(samples_per_shard):
            curve = None

            # Continue until the model is valid
            while True:
                # Sample a seismic model
                model = sample_model(
                    prior=prior,
                    params=params,
                    random_seed=np.random.randint(0, 2**32 - 1),
                    sort_vs=False
                )

                # Select a minimum period for the dispersion curve between min_p and (max_p - min_period_difference)
                if variable_grid:
                    # Select a random min. period
                    min_period = np.random.uniform(min_p, max_p - min_period_difference)

                    # Select a random range of periods between min_period_difference and (max_p - min_p)
                    period_range = np.random.uniform(
                        min_period_difference,
                        max_p - min_period
                    )

                    # Max period is min_period + period_range
                    max_period = min_period + period_range
                else:
                    min_period = min_p
                    max_period = max_p
                # end if

                try:
                    # Forward simulation to generate dispersion curve
                    curve = model.forward(
                        length=length,
                        min_p=min_period,
                        max_p=max_period
                    )
                    break
                except ValueError as e:
                    continue
                # end if
            # end while

            # Package the model and curve
            sample = SeismicSample(model, curve)
            model_sample = sample.to_arrow_dict()

            # Transform the model to layers profile
            model_curve = vornoi_to_layers(
                vs=np.array(sample.to_arrow_dict()['vs']),
                z=np.array(sample.to_arrow_dict()['z']),
                z_max=int(prior.z[1]),
                n_points=60
            )

            # Add the model profile to the sample
            model_sample['model_profile'] = model_curve
            samples.append(model_sample)
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
        console.log(
            "You can now upload the dataset on HuggingFace with: "
            "`huggingface-cli upload MIGRATE/<dataset-name> <loca-path> . --repo-type dataset "
            "--commit-message \"<commit-message>\" --private`"
        )
    # end if

    # Save dataset info
    if write_dataset_info:
        assert description is not None, "Description is required for dataset card generation."
        assert name is not None, "Name is required for dataset card generation."
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
        assert pretty_name is not None, "Pretty name is required for dataset card generation."
        assert license_name is not None, "License name is required for dataset card generation."

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
