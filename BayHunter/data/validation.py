#
# MIGRATE
#

# Imports
import numpy as np
from BayHunter.data import SeismicModel, SeismicPrior, SeismicParams


# Validate a model
def validate_model(
        model: SeismicModel,
        prior: SeismicPrior,
        params: SeismicParams
):
    """
    Validate a model.

    :param params: Parameters for the model
    :param prior: Prior distribution
    :param model: Model to validate
    :return: True if model is valid, False otherwise
    """
    # Get Vp, Vs and h from the model
    vp, vs, h = model.get_vp_vs_h(
        prior.vpvs,
        prior.mantle
    )

    # Check wether number of layer lies within the prior
    layer_min = prior.layers[0]
    layer_max = prior.layers[1]
    if model.nlayers < layer_min or model.nlayers > layer_max:
        return False
    # end if

    # Check model for layer with thickness of smaller than thickmin
    if np.any(h[:-1] < params.thickmin):
        return False
    # end if

    # Check whether Vs lies within the prior
    vs_min = prior.vs[0]
    vs_max = prior.vs[1]
    if np.any(vs < vs_min) or np.any(vs > vs_max):
        return False
    # end if

    # Check whether interfaces lie within the prior
    z_min = prior.z[0]
    z_max = prior.z[1]
    z = np.cumsum(h)
    if np.any(z < z_min) or np.any(z > z_max):
        return False
    # end if

    # Check model for low velocity zones.
    # If larger than perc, then compvels must be positive
    if params.lvz is not None:
        compvels = vs[1:] - (vs[:-1] * (1 - params.lvz))
        if not compvels.size == compvels[compvels > 0].size:
            return False
        # end if
    # end if
# end validate_model

