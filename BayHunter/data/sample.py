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
):
    """
    Sample a seismic model.

    :param vpvs: Vp/Vs ratio
    :param prior: Prior distribution
    :param params: Parameters for the model
    :param random_state: Random state for reproducibility
    :return: Sampled model as a SeismicModel object
    """
    # Get min and max values for each parameter
    z_min, z_max = prior.z
    vs_min, vs_max = prior.vs
    layers_min, layers_max = prior.layers

    # Sample the number of layers
    n_layers = np.random.randint(layers_min, layers_max + 1)

    # Sample Vs
    vs = random_state.uniform(low=vs_min, high=vs_max, size=n_layers)
    vs = np.sort(vs)

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


# Sample a model from the prior
def sample_model(
        prior: SeismicPrior,
        params: SeismicParams,
        random_seed: int = 42
) -> SeismicModel:
    """
    Sample a model from the prior.

    :param prior: Prior distribution
    :param params: Parameters for the model
    :param random_seed: Random seed for reproducibility
    :return: Sampled model as a SeismicModel object
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
        random_state=random_state
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

