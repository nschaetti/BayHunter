#
# MIGRATE
#

import os
import numpy as np
import click
from rich.console import Console
from rich.table import Table
import configparser

from BayHunter.data import SurfDispModel
from BayHunter.data import SeismicPrior, sample_model, SeismicParams


console = Console()


@click.group()
def cli():
    pass
# end cli


@cli.command("sample-model")
@click.option("--ini-file", type=click.Path(exists=True), required=True, help="Path to .ini file containing modelpriors")
@click.option("--random-seed", type=int, default=42, help="Random seed for reproducibility")
def sample_model_cli(ini_file, random_seed):
    """
    Sample a model from a prior specified in an .ini file and print it.
    """
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
        random_seed=random_seed
    )

    # Show the sampled model
    click.echo("Sampled Seismic Model:")
    click.echo(model)
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
        period_length: int = 60,
        reference='rdispgr'
):
    """
    Run the surfdisp96 model.

    :param h: Layer thicknesses (km)
    :param vp: P-wave velocities (km/s)
    :param vs: S-wave velocities (km/s)
    :param rho: Layer densities (g/cm³)
    :param period_min: Minimum period (s)
    :param period_max: Maximum period (s)
    :param reference: Reference model for dispersion calculation
    :return: Dispersion velocities (km/s)
    """
    # Run the model
    model = SurfDispModel(
        obsx=np.linspace(period_min, period_max, period_length),
        ref=reference,
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
