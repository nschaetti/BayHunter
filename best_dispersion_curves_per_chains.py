
# Imports
import os
import argparse
from scipy.io import loadmat
import numpy as np
import os.path as op
from rich import print
from rich.traceback import install
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import cm
import glob

from BayHunter import Model
from BayHunter import Targets


# Activate Rich
install(show_locals=True)


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

    return xmod, ymod, yobs, cdepth, cvs
# end get_disp_curve


def load_data(
        targets,
        directory: str,
        n_chains: int,
        top_n: int = 25
):
    """
    Load vpvs data from .mat files.

    Args:
        targets (Targets): Targets.
        directory (str): Directory containing .mat files.
        n_chains (int): Number of chains.
        top_n (int): Number of top models to consider.
    """
    # Initialize data
    data = []
    chain_misfits = []
    models_vs_z_all = []

    # For each chain
    for i in range(n_chains):
        # Load models, misfits and vpvs
        models = np.load(op.join(directory, f"c{i:03d}_p2models.npy"))
        misfits = np.load(op.join(directory, f"c{i:03d}_p2misfits.npy"))[:,0]
        vpvs = np.load(op.join(directory, f"c{i:03d}_p2vpvs.npy"))

        # Sanity check
        if len(misfits) < top_n:
            top_n = len(misfits)
        # end if

        # Get best
        best_indices = np.argsort(misfits)[:top_n]

        # Select corresponding data
        top_misfits = misfits[best_indices]
        top_models = models[best_indices]
        top_vpvs = vpvs[best_indices]

        # Compute dispersion curve for each model
        top_disp_curves = []
        top_vs_z = []
        for i, model in enumerate(top_models):
            xmod, ymod, yobs, cdepth, cvs = get_disp_curve(targets, model, top_vpvs[i])
            disp_curves = np.concatenate((xmod.reshape((1, -1)), ymod.reshape((1, -1))), 0).reshape(1, 2, -1)
            top_disp_curves.append(disp_curves)
            vs_z_pair = np.stack([cvs, cdepth], axis=0)  # shape (2, variable_depth)
            top_vs_z.append(vs_z_pair)
        # end for
        data_disp_curves = np.concatenate(top_disp_curves, axis=0).reshape(1, top_n, 2, -1)
        data.append(data_disp_curves)
        chain_misfits.append(top_misfits)
        models_vs_z_all.append(top_vs_z)  # list of lists of arrays (top_n, 2, variable_depth)
    # end for

    return np.concatenate(data, axis=0), np.array(chain_misfits), models_vs_z_all
# end load_data


def plot_dispersion_curves(
        T,
        vg,
        targets,
        output_directory: str,
        directory: str,
        n_chains: int,
        top_n: int = 25
):
    """
    Plot dispersion curves from .mat files.
    """
    # Load data
    disp_curves, misfits, top_vs_z = load_data(
        targets=targets,
        directory=directory,
        n_chains=n_chains,
        top_n=top_n
    )

    # Print shapes
    print(f"disp_curves.shape = {disp_curves.shape}")
    print(f"misfits.shape = {misfits.shape}")

    n_chains, top_k, _, n_periods = disp_curves.shape
    cmap = get_cmap("tab10")  # Up to 10 distinct base colors

    plt.figure(figsize=(12, 6))

    saved_misfits = ""
    for chain_idx in range(n_chains):
        base_color = cmap(chain_idx % 10)
        mean_misfit = np.mean(misfits[chain_idx])
        saved_misfits += f"{np.mean(misfits[chain_idx])}\n"
        for model_idx in range(top_k):
            x = disp_curves[chain_idx, model_idx, 0]
            y = disp_curves[chain_idx, model_idx, 1]
            alpha = 0.4 + 0.6 * (1 - model_idx / top_k)  # More opaque for best models
            plt.plot(x, y, color=base_color, alpha=alpha)
        # end for

        # Légende : numéro de chaîne + misfit moyen
        label = f"Chain {chain_idx} (mean misfit = {mean_misfit:.2f})"
        plt.plot([], [], color=base_color, label=label)
    # end for

    # Plot observed
    plt.plot(T, vg, color='black', linewidth=2.5, label='Observed')

    plt.xlabel("Period (s)")
    plt.ylabel("Phase velocity (km/s)")
    plt.title("Top 25 Dispersion Curves per MCMC Chain")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    print(f"Saving dispersion curves to {op.join(output_directory, 'dispersion_curves.png')}")
    plt.savefig(op.join(output_directory, "dispersion_curves.png"), dpi=300)

    # Save misfits
    print(f"Saving misfits to {op.join(output_directory, 'misfits.txt')}")
    with open(op.join(output_directory, "misfits.txt"), "w") as f:
        f.write(saved_misfits)
    # end with

    fig, axes = plt.subplots(1, n_chains, figsize=(4 * n_chains, 8), sharey=True)

    if n_chains == 1:
        axes = [axes]

    for chain_idx in range(n_chains):
        chain_models = top_vs_z[chain_idx]
        base_color = cmap(chain_idx % 10)
        ax = axes[chain_idx]

        layer_counts = []
        for vs, depths in chain_models:
            ax.plot(vs, depths, color=base_color, alpha=0.6)
            layer_counts.append(len(vs))

        mean_layers = np.mean(layer_counts)
        ax.invert_yaxis()
        ax.set_xlabel("Vs (km/s)")
        if chain_idx == 0:
            ax.set_ylabel("Depth (km)")
        ax.set_title(f"Chain {chain_idx} ({mean_layers:.1f} layers)")
        ax.grid(True)

    plt.tight_layout()
    # plt.show()
    print(f"Saving vs(z) curves to {op.join(output_directory, 'vs_z_curves.png')}")
    plt.savefig(op.join(output_directory, "vs_z_curves.png"), dpi=300)
# end plot_dispersion_curves


# Main function
def main():
    parser = argparse.ArgumentParser(description="Plot BayHunter p2likes for all chains")
    parser.add_argument("--mat-file", type=str, help="Path to the .mat file")
    parser.add_argument("--directory", type=str, help="Directory containing c*_p1likes.npy files")
    parser.add_argument("--n-chains", type=int, default=8, help="Number of chains")
    parser.add_argument("--top-n", type=int, default=25, help="Top N models to consider")
    parser.add_argument("--output-directory", type=str, help="Output directory for plots")
    args = parser.parse_args()

    # Loading the .mat file
    data = loadmat(args.mat_file)
    T = data['T_pick_interp'].squeeze()[::-1]
    vg = data['vg_pick_interp'].squeeze()[::-1] / 1000.0
    target_swd = Targets.RayleighDispersionPhase(T, vg)
    targets = Targets.JointTarget(targets=[target_swd])

    # Plot dispersion curves
    plot_dispersion_curves(
        T=T,
        vg=vg,
        targets=targets,
        directory=args.directory,
        output_directory=args.output_directory,
        n_chains=args.n_chains,
        top_n=args.top_n
    )
# end main


if __name__ == "__main__":
    main()
# end if


