#
# MIGRATE
#

# Imports
from typing import Dict, Optional, Union, Tuple
import numpy as np

from .validation import validate_model
from .model import SeismicModel, SeismicPrior, SeismicParams


# Sample VpVs ratio
def sample_vpvs(
        vpvs: Union[float, Tuple[float, float]],
        random_state: np.random.RandomState
):
    """
    Sample VpVs ratio.

    :param vpvs: VpVs ratio or range of VpVs ratios
    :param random_state: Random state for reproducibility
    :return:
    """
    if type(vpvs) == float:
        return vpvs
    # end if
    vpvsmin, vpvsmax = vpvs
    return random_state.uniform(low=vpvsmin, high=vpvsmax)
# end sample_vpvs


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
    prior.vpvs = sample_vpvs(prior.vpvs, random_state)

    # Get min and max values for each parameter
    z_min, z_max = prior.z
    vs_min, vs_max = prior.vs
    layers_min, layers_max = prior.layers

    # Sample the number of layers
    n_layers = np.random.randint(layers_min, layers_max + 1)

    # Sample Vs
    vs = random_state.uniform(low=vs_min, high=vs_max, size=n_layers)

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
        if n_layers  == 2:
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
    np.sort(z_vnoi)

    # Put Vs and layers together
    model = SeismicModel(np.concatenate((vs, z_vnoi)))

    # Check the model
    if validate_model(model, prior, params):
        return model
    else:
        return sample_model(prior, params, random_seed=random_seed+1)
    # end if
# end sample_model

