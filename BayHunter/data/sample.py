#
# MIGRATE
#

# Imports
from typing import Dict, Optional, Union, Tuple
import numpy as np
from rich.console import Console

from .validation import validate_model
from .model import SeismicModel, SeismicPrior, SeismicParams


console = Console()


# Sample VpVs ratio
def sample_vpvs(
        vpvs: Union[float, Tuple[float, float]],
        random_state: np.random.RandomState
):
    """
    Sample VpVs ratio.

    :param vpvs: VpVs ratio or range of VpVs ratios
    :param random_state: Random state for reproducibility
    :return: Sampled VpVs ratio
    """
    if type(vpvs) == float:
        return vpvs
    # end if
    vpvsmin, vpvsmax = vpvs
    return random_state.uniform(low=vpvsmin, high=vpvsmax)
# end sample_vpvs


# Sample noise
def sample_noise(
        prior: SeismicPrior,
        random_state: np.random.RandomState
):
    """
    Sample noise.

    :param prior: Prior distribution
    :param random_state: Random state for reproducibility
    :return: Sampled noise
    """
    noiserefs = ['noise_corr', 'noise_sigma']
    init_noise = np.ones(2) * np.nan
    corrfix = np.zeros(2, dtype=bool)

    noise_priors = []
    for j, noiseref in enumerate(noiserefs):
        noiseprior = getattr(prior, noiseref)
        if type(noiseprior) in [int, float, np.floating]:
            corrfix[j] = True
            init_noise[j] = noiseprior
        else:
            init_noise[j] = random_state.uniform(low=noiseprior[0], high=noiseprior[1])
        # end if
        noise_priors.append(noiseprior)
    # end for

    # Check if all noise parameters are fixed
    noiseinds = np.where(corrfix == 0)[0]
    if len(noiseinds) == 0:
        console.print('[orange]All your noise parameters are fixed. On Purpose?[/orange]')
    # end if

    return init_noise, corrfix, noise_priors
# end sample_noise


# Sample seismic model
def sample_seismic_model(
        vpvs: float,
        prior: SeismicPrior,
        params: SeismicParams,
        random_state: np.random.RandomState,
        sort_vs: bool = True
):
    """
    Sample a seismic model.

    Args:
        vpvs (float): VpVs ratio
        prior (SeismicPrior): Prior distribution
        params (SeismicParams): Parameters for the model
        random_state (np.random.RandomState): Random state for reproducibility
        sort_vs (bool): Whether to sort layers or not
    """
    # Get min and max values for each parameter
    z_min, z_max = prior.z
    vs_min, vs_max = prior.vs
    layers_min, layers_max = prior.layers

    # Sample the number of layers
    n_layers = np.random.randint(layers_min, layers_max + 1)

    # Sample Vs
    vs = random_state.uniform(low=vs_min, high=vs_max, size=n_layers)

    # Sort the Vs
    if sort_vs:
        vs = np.sort(vs)
    # end if

    # Mohoest ?
    if prior.mohoest is not None and n_layers > 1:
        # Sample the moho depth
        moho_mean, moho_std = prior.mohoest

        # Sample the moho depth
        moho_depth = random_state.normal(loc=moho_mean, scale=moho_std)

        # Sample the depth of the layers
        z_tmp = random_state.uniform(1, np.min([5, moho_depth]))
        z_vnoi_tmp = [moho_depth - z_tmp, moho_depth + z_tmp]

        # Add layers
        if n_layers == 2:
            z_vnoi = z_vnoi_tmp
        else:
            z_vnoi = np.concatenate(
                [
                    z_vnoi_tmp,
                    random_state.uniform(low=z_min, high=z_max, size=n_layers - 2)
                ]
            )
        # end if
    else:
        z_vnoi = random_state.uniform(low=z_min, high=z_max, size=n_layers)
    # end if

    # Sort the layers
    z_vnoi = np.sort(z_vnoi)

    # Put Vs and layers together
    model = SeismicModel(
        model=np.concatenate((vs, z_vnoi)),
        vpvs=vpvs
    )

    # Check the model
    if validate_model(model, prior, params):
        return model
    else:
        return sample_seismic_model(
            vpvs=vpvs,
            prior=prior,
            params=params,
            random_state=random_state
        )
    # end if
# end sample_seismic_model


# Transform Vornoi Kernels to layers curve
def vornoi_to_layers(
        vs: np.ndarray,
        z: np.ndarray,
        z_max: int,
        n_points: int
):
    """
    Transform Voronoi kernels to a layered profile using NumPy.

    Args:
        vs (np.ndarray): S-wave velocities of each Voronoi kernel (1D).
        z (np.ndarray): Depths (in m) of each Voronoi kernel (1D).
        z_max (float): Maximum depth of the model.
        n_points (int): Number of points in the final profile.

    Returns:
        tuple: (vs_profile, z_profile) both of shape (n_points,)
    """
    # Checks
    assert vs.ndim == 1 and z.ndim == 1, "Inputs must be 1D arrays"
    assert vs.shape[0] == z.shape[0], "vs and z must have same length"
    assert np.all(z >= 0), "Depths must be non-negative"

    # Remove zero-depth points (padding)
    mask = z > 0
    z = z[mask]
    vs = vs[mask]

    # Discontinuities (mid-depths between Voronoi zones)
    mid_z = (z[:-1] + z[1:]) / 2

    # Add surface (0 m)
    mid_z = np.concatenate(([0.0], mid_z))

    # Clip to z_max
    mask = mid_z <= z_max
    mid_z = mid_z[mask]
    vs = vs[mask]

    # Add z_max at the end if needed
    if mid_z[-1] != z_max:
        mid_z = np.concatenate((mid_z, [z_max]))
        vs = np.concatenate((vs, [vs[-1]]))  # Extend with last velocity

    # Compute vertical resolution
    dz = z_max / n_points

    # Output profiles
    output_vs = np.zeros(n_points, dtype=np.float32)
    output_z = np.zeros(n_points, dtype=np.float32)

    # Fill in the layered model
    for i in range(1, len(mid_z)):
        start_idx = int(mid_z[i - 1] / dz)
        end_idx = int(mid_z[i] / dz)
        output_vs[start_idx:end_idx] = vs[i - 1]
        output_z[start_idx:end_idx] = (np.arange(start_idx, end_idx) + 0.5) * dz
    # end for

    return output_vs, output_z
# end vornoi_to_layers


# Sample a model from the prior
def sample_model(
        prior: SeismicPrior,
        params: SeismicParams,
        random_seed: int = 42,
        sort_vs: bool = True
) -> SeismicModel:
    """
    Sample a model from the prior.

    Args:
        prior (SeismicPrior): Prior distribution
        params (SeismicParams): Parameters for the model
        random_seed (int): Random seed for reproducibility
        sort_vs (bool): Whether to sort layers or not
    """
    # Random state
    random_state = np.random.RandomState(random_seed)

    # Sample VpVs ratio
    vpvs = sample_vpvs(prior.vpvs, random_state)

    # Sample seismic model
    model = sample_seismic_model(
        vpvs=vpvs,
        prior=prior,
        params=params,
        random_state=random_state,
        sort_vs=sort_vs
    )

    return model
# end sample_model


# Forward process of a model
def forward_modeling():
    """
    Forward process of a model.
    """
    pass
# end forward_modeling

