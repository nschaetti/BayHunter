
# Imports
import os
import argparse
from scipy.io import loadmat
import numpy as np
import os.path as op
from rich import print
from rich.traceback import install
import matplotlib.pyplot as plt
from matplotlib import cm


# Activate Rich
install(show_locals=True)

# BayHunter imports
from BayHunter import utils
from BayHunter import SynthObs
from BayHunter import Model
from BayHunter import Targets
from BayHunter import MCMC_Optimizer
from BayHunter import PlotFromStorage


def float_or_tuple(text):
    if text.lower() == 'none':
        return None
    parts = text.split(',')
    if len(parts) == 1:
        return float(parts[0])
    return tuple(map(float, parts))


def tuple_of_floats(text):
    return tuple(map(float, text.split(',')))
# end tuple_of_floats


def float_or_none(text):
    return None if text.lower() == 'none' else float(text)
# end float_or_none


def override_params(
        args,
        priors,
        initparams
):
    """
    Override parameters from CLI arguments.

    Args:
        args_dict (dict): Arguments from CLI.
        priors (dict): Priors.
        initparams (dict): Initial parameters.

    Returns:
        tuple: Tuple containing the updated priors and initial parameters.
    """
    # Create dict from args
    args_dict = vars(args)

    # mapping des arguments CLI → sections .ini
    priors_keys = [
        "vpvs", "layers", "vs", "z", "mohoest",
        "rfnoise_corr", "swdnoise_corr", "rfnoise_sigma", "swdnoise_sigma"
    ]
    initparams_keys = [
        "nchains", "iter_burnin", "iter_main", "propdist", "acceptance",
        "thickmin", "lvz", "hvz", "rcond", "station", "savepath", "maxmodels"
    ]

    # Override priors
    for key in priors_keys:
        argname = key.replace('_', '-') if '-' in key else key
        value = args_dict.get(argname.replace('-', '_'))
        if value is not None:
            priors[key] = value
        # end if
    # end for

    for key in initparams_keys:
        argname = key.replace('_', '-') if '-' in key else key
        value = args_dict.get(argname.replace('-', '_'))
        if value is not None:
            initparams[key] = value
        # end if
    # end for

    return priors, initparams
# end override_params


# Get dispersion curve
def get_disp_curve(
        target,
        model,
        vpvs,
        mantle = None
):
    """
    Get the dispersion curve from a model.

    Args:
        model (Model): Model.
        vpvs (float): Vp/Vs ratio.
    """
    # Get model parameters
    vp, vs, h = Model.get_vp_vs_h(model, vpvs, mantle)
    rho = vp * 0.32 + 0.77

    # Get step model
    cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)

    # Compute model
    xmod, ymod = target.targets[0].moddata.plugin.run_model(
        h=h,
        vp=vp,
        vs=vs,
        rho=rho
    )

    # Observations
    yobs = target.targets[0].obsdata.y

    return xmod, ymod, yobs, cdepth
# end get_disp_curve


def compute_top_misfit_stats(
        targets,
        savepath,
        top_n=100
):
    """
    Computes the mean, std, and min of the top N misfit values from a BayHunter run.

    Parameters:
    - savepath (str): Path to the BayHunter results directory (should contain 'c_misfits.npy')
    - top_n (int): Number of best (lowest) misfits to consider

    Returns:
    - dict with keys: mean, std, min, count
    """
    misfit_file = os.path.join(savepath, 'data', 'c_misfits.npy')
    model_file = os.path.join(savepath, 'data', 'c_models.npy')
    vpvs_file = os.path.join(savepath, 'data', 'c_vpvs.npy')

    for f in [misfit_file, model_file, vpvs_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Could not find {f}")
        # end if
    # end if

    # Load data
    misfits = np.load(misfit_file)[:,0]
    models = np.load(model_file)
    vpvs = np.load(vpvs_file)

    # Sanity check
    if len(misfits) < top_n:
        top_n = len(misfits)
    # end if

    # Get 100 best
    best_indices = np.argsort(misfits)[:top_n]

    # Select corresponding data
    top_misfits = misfits[best_indices]
    top_models = models[best_indices]
    top_vpvs = vpvs[best_indices]

    # best_indices: (100,)
    # top_misfits: (100,)
    # top_models: (100, 42)
    # top_vpvs: (100,)

    # Compute dispersion curve for each model
    top_disp_curves = []
    for i, model in enumerate(top_models):
        xmod, ymod, yobs, cdepth = get_disp_curve(targets, model, top_vpvs[i])
        top_disp_curves.append(
            np.concatenate((xmod.reshape((1, -1)), ymod.reshape((1, -1))), 0).reshape(1, 2, -1)
        )
    # end for
    top_disp_curves = np.concatenate(top_disp_curves, axis=0)

    return {
        'stats': {
            'mean': np.mean(top_misfits),
            'std': np.std(top_misfits),
            'min': np.min(top_misfits),
            'count': top_n
        },
        'indices': best_indices,
        'models': top_models,
        'vpvs': top_vpvs,
        'disp_curves': top_disp_curves,
        'yobs': yobs
    }
# end compute_top_misfit_stats


# Plot dispersion curve and observation
def plot_disp_curve(models, disp_curves, yobs):
    """
    Plot dispersion curve and observation.
    """
    # Number of models
    n_models = models.shape[0]

    # Création de la colormap
    colors = cm.viridis(np.linspace(0, 1, n_models))

    # Plot 1 : courbes de dispersion
    plt.figure(figsize=(10, 6))
    for i in range(n_models):
        T = disp_curves[i, 0]
        V = disp_curves[i, 1]
        plt.plot(T, V, color=colors[i], alpha=0.6)
    # end for

    plt.plot(T, yobs, color='black', linewidth=2.5, label='Observed')
    plt.xlabel("Period (s)")
    plt.ylabel("Phase velocity (km/s)")
    plt.title("Dispersion curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return colors
# end plot_disp_curve


def plot_models(models, colors, dz=1.0):
    """
    Plot Vs(z) models from a 2D array of shape (n_models, n_params),
    where each row contains [vs1, vs2, ..., vsN, z1, z2, ..., zN] with possible NaNs.
    If no depths are available, uses np.arange with dz as step.

    Parameters:
    - models: ndarray, shape (n_models, n_params), with NaNs for unused layers
    - colors: list of matplotlib colors of length n_models
    - dz: float, step in km for synthetic depths if not available
    """
    n_models = models.shape[0]
    for i in range(n_models):
        model = models[i]
        model = model[~np.isnan(model)]
        n = len(model) // 2
        vs = model[:n]
        depths = np.arange(n) * dz
        plt.plot(vs, depths, color=colors[i], alpha=0.6)
    # end for

    plt.gca().invert_yaxis()
    plt.xlabel("Vs (km/s)")
    plt.ylabel("Depth (km)")
    plt.title("Vs models (using synthetic depth)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# end plot_models_from_array


def main(args):
    # Loading the .mat file
    print("[bold cyan]Loading realworld data...[/bold cyan]")
    data = loadmat(args.mat_file)

    # Extract period and group velocity
    T = data['T_pick_interp'].squeeze()[::-1]  # Periods
    vg = data['vg_pick_interp'].squeeze()[::-1] / 1000.0  # Group velocity

    # Check
    print(f"Periods (T): {T.shape}: {T}")
    print(f"Speed (vg): {vg.shape}: {vg}")

    # Title: Inversion of Rayleigh Dispersion Phase and Receiver Function
    print("[bold cyan]Initializing targets...[/bold cyan]")
    print(f"[yellow]RayleighDispersionPhase: T:{T.size}, vg:{vg.size} points[/]")

    print("[bold cyan]Initializing RayleighDispersionPhase target...[/bold cyan]")
    target_swd = Targets.RayleighDispersionPhase(T, vg)

    # Join the targets
    targets = Targets.JointTarget(targets=[target_swd])

    # Load parameters from config.ini
    print("[bold cyan]Loading parameters from config.ini...[/bold cyan]")
    priors, initparams = utils.load_params('tutorial/config.ini')

    # Create dict from args
    priors, initparams = override_params(args, priors, initparams)

    print(f"[yellow]Priors: {priors}[/]")
    print(f"[yellow]Initparams: {initparams}[/]")

    print("[bold cyan]Saving config for BayWatch...[/bold cyan]")
    utils.save_baywatch_config(
        targets=targets,
        priors=priors,
        initparams=initparams
    )

    # MCMC inversion using the MCMC_Optimizer class
    optimizer = MCMC_Optimizer(
        targets=targets,
        initparams=initparams,
        priors=priors,
        random_seed=None
    )

    print("[bold green]Starting MCMC inversion...[/bold green]")
    optimizer.mp_inversion(
        nthreads=8,
        baywatch=True,
        dtsend=1
    )

    print("[bold cyan]Post-processing results...[/bold cyan]")
    path = initparams['savepath']
    cfile = f"{initparams['station']}_config.pkl"
    configfile = op.join(path, 'data', cfile)
    obj = PlotFromStorage(configfile=configfile)

    # print("[bold cyan]Saving posterior distributions and plots...[/bold cyan]")
    obj.save_final_distribution(maxmodels=100000, dev=0.05)
    obj.save_plots()
    obj.merge_pdfs()

    # Get 100 best
    res = compute_top_misfit_stats(targets=targets, savepath=path, top_n=100)

    # Stats: {'mean': 0.26789723575115204, 'std': 9.025763284506014e-06, 'min': 0.26788872480392456, 'count': 100}
    # Top models: (100, 42)
    # Top dispersion curves: (100, 2, 108)

    # Plot dispersion curve and observation
    colors = plot_disp_curve(res['models'], res['disp_curves'], res['yobs'])

    # Plot models
    plot_models(res['models'], colors)

    print("[bold green]✅ Done! Results saved in:[/bold green]", path)
# end main


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description="Inversion of Rayleigh Dispersion Phase and Receiver Function"
    )

    # Input data
    parser.add_argument("--mat-file", type=str, help="Path to the .mat file")

    # modelpriors
    parser.add_argument("--vpvs", type=float_or_tuple, help="Vp/Vs ratio or range")
    parser.add_argument("--layers", type=tuple_of_floats, help="Min,max number of layers")
    parser.add_argument("--vs", type=tuple_of_floats, help="Min,max Vs (km/s)")
    parser.add_argument("--z", type=tuple_of_floats, help="Min,max depth (km)")
    parser.add_argument("--mohoest", type=float_or_none, help="Moho estimate or None")
    parser.add_argument("--rfnoise-corr", type=float, help="RF noise correlation")
    parser.add_argument("--swdnoise-corr", type=float, help="SWD noise correlation")
    parser.add_argument("--rfnoise-sigma", type=tuple_of_floats, help="RF noise std range")
    parser.add_argument("--swdnoise-sigma", type=tuple_of_floats, help="SWD noise std range")

    # initparams
    parser.add_argument("--nchains", type=int, help="Number of MCMC chains")
    parser.add_argument("--iter-burnin", type=int, help="Number of burn-in iterations")
    parser.add_argument("--iter-main", type=int, help="Number of main iterations")
    parser.add_argument("--propdist", type=tuple_of_floats, help="Proposal distributions")
    parser.add_argument("--acceptance", type=tuple_of_floats, help="Acceptance rate range")
    parser.add_argument("--thickmin", type=float, help="Minimum thickness of layers")
    parser.add_argument("--lvz", type=float_or_none, help="Low velocity zone constraint")
    parser.add_argument("--hvz", type=float_or_none, help="High velocity zone constraint")
    parser.add_argument("--rcond", type=float, help="rcond for singular values")
    parser.add_argument("--station", type=str, help="Station name")
    parser.add_argument("--savepath", type=str, help="Save path for results")
    parser.add_argument("--maxmodels", type=int, help="Max models saved per chain")

    args = parser.parse_args()

    main(args)
# End of file
