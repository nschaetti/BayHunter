#
# MIGRATE
# Copyright Nils Schaetti
#

from typing import Optional, Tuple
import numpy as np
from BayHunter.surfdisp96_ext import surfdisp96


# Integer values
KMAX_MAX = 60

# Kmax override error
ERROR_KMAX_OVERRIDE_STRING = "Your observed data vector exceeds the maximum of 60 \
periods that is allowed in SurfDisp. For forward modeling SurfDisp will \
reduce the samples to 60 by linear interpolation within the given period \
span.\nFrom this data, the dispersion velocities to your observed periods \
will be determined. The precision of the data will depend on the distribution \
of your samples and the complexity of the input velocity-depth model."


class SeismicParams:
    """
    Configuration parameters used during inversion or validation of seismic models.
    """

    def __init__(
        self,
        nchains: int = 5,
        iter_burnin: int = 2048 * 16,
        iter_main: int = 2048 * 8,
        propdist: Tuple[float, float, float, float, float] = (0.015, 0.015, 0.015, 0.005, 0.005),
        acceptance: Tuple[float, float] = (40, 45),
        thickmin: float = 0.1,
        lvz: Optional[float] = None,
        hvz: Optional[float] = None,
        rcond: float = 1e-5,
        station: str = "test",
        savepath: str = "results",
        maxmodels: int = 50000,
    ):
        """
        Constructor for SeismicParams.

        :param nchains: Number of chains for MCMC sampling.
        :param iter_burnin: Number of burn-in iterations.
        :param iter_main: Number of main iterations.
        :param propdist: Proposed distribution for MCMC sampling.
        :param acceptance: Acceptance rates for MCMC sampling.
        :param thickmin: Minimum thickness of layers.
        :param lvz: Low velocity zone parameters.
        :param hvz: High velocity zone parameters.
        :param rcond: Regularization condition.
        :param station: Station name.
        :param savepath: Path to save results.
        :param maxmodels: Maximum number of models to sample.
        """
        self.nchains = nchains
        self.iter_burnin = iter_burnin
        self.iter_main = iter_main
        self.propdist = propdist
        self.acceptance = acceptance
        self.thickmin = thickmin
        self.lvz = lvz
        self.hvz = hvz
        self.rcond = rcond
        self.station = station
        self.savepath = savepath
        self.maxmodels = maxmodels
    # end __init__

    @classmethod
    def from_dict(cls, d: dict) -> "SeismicParams":
        """
        Create SeismicParams from a dictionary.

        :param d: Dictionary containing parameters.
        :return: SeismicParams object.
        """
        def parse(val):
            if val is None or val == "None":
                return None
            if isinstance(val, str) and "," in val:
                return tuple(float(x.strip()) for x in val.split(","))
            try:
                return int(val) if str(val).isdigit() else float(val)
            except Exception:
                return val
            # end try
        # end parse

        return cls(
            nchains=int(d.get("nchains", 5)),
            iter_burnin=int(eval(d.get("iter_burnin", "2048 * 16"))),
            iter_main=int(eval(d.get("iter_main", "2048 * 8"))),
            propdist=parse(d.get("propdist", "0.015, 0.015, 0.015, 0.005, 0.005")),
            acceptance=parse(d.get("acceptance", "40, 45")),
            thickmin=float(d.get("thickmin", 0.1)),
            lvz=parse(d.get("lvz", None)),
            hvz=parse(d.get("hvz", None)),
            rcond=float(d.get("rcond", 1e-5)),
            station=d.get("station", "test"),
            savepath=d.get("savepath", "results"),
            maxmodels=int(d.get("maxmodels", 50000)),
        )
    # end from_dict

    def __str__(self):
        """
        String representation of the SeismicParams object.

        :return: String representation
        """
        return (
            f"SeismicParams(nchains={self.nchains}, iter_burnin={self.iter_burnin}, iter_main={self.iter_main}, "
            f"propdist={self.propdist}, acceptance={self.acceptance}, thickmin={self.thickmin}, "
            f"lvz={self.lvz}, hvz={self.hvz}, rcond={self.rcond}, station='{self.station}', "
            f"savepath='{self.savepath}', maxmodels={self.maxmodels})"
        )
    # end __str__

# end SeismicParams


class SeismicPrior:
    """
    Prior configuration for seismic model sampling.
    """

    def __init__(
        self,
        vs: Tuple[float, float],
        z: Tuple[float, float],
        layers: Tuple[int, int],
        vpvs: float = 1.73,
        mohoest: Optional[Tuple[float, float]] = None,
        mantle: Optional[Tuple[float, float]] = None,
        rfnoise_corr: Optional[float] = 0.9,
        swdnoise_corr: Optional[float] = 0.0,
        rfnoise_sigma: Optional[Tuple[float, float]] = (1e-5, 0.05),
        swdnoise_sigma: Optional[Tuple[float, float]] = (1e-5, 0.05),
    ):
        """
        Constructor for SeismicPrior.
        """
        self.vs = vs
        self.z = z
        self.layers = layers
        self.vpvs = vpvs
        self.mohoest = mohoest
        self.mantle = mantle
        self.rfnoise_corr = rfnoise_corr
        self.swdnoise_corr = swdnoise_corr
        self.rfnoise_sigma = rfnoise_sigma
        self.swdnoise_sigma = swdnoise_sigma
    # end __init__

    @classmethod
    def from_dict(cls, d: dict) -> "SeismicPrior":
        def parse(val):
            if val is None or val == 'None':
                return None
            if isinstance(val, str) and "," in val:
                parts = [float(x.strip()) for x in val.split(",")]
                return tuple(parts)
            try:
                return float(val)
            except Exception:
                return val
        # end parse

        return cls(
            vs=parse(d.get("vs", (2.0, 5.0))),
            z=parse(d.get("z", (0.0, 60.0))),
            layers=parse(d.get("layers", (1, 20))),
            vpvs=parse(d.get("vpvs", 1.73)),
            mohoest=parse(d.get("mohoest", None)),
            mantle=parse(d.get("mantle", None)),
            rfnoise_corr=parse(d.get("rfnoise_corr", 0.9)),
            swdnoise_corr=parse(d.get("swdnoise_corr", 0.0)),
            rfnoise_sigma=parse(d.get("rfnoise_sigma", (1e-5, 0.05))),
            swdnoise_sigma=parse(d.get("swdnoise_sigma", (1e-5, 0.05))),
        )
    # end from_dict

    def __str__(self):
        return (
            f"SeismicPrior(vs={self.vs}, z={self.z}, layers={self.layers}, "
            f"vpvs={self.vpvs}, mohoest={self.mohoest}, mantle={self.mantle}, "
            f"rfnoise_corr={self.rfnoise_corr}, swdnoise_corr={self.swdnoise_corr}, "
            f"rfnoise_sigma={self.rfnoise_sigma}, swdnoise_sigma={self.swdnoise_sigma})"
        )
    # end __str__

# end SeismicPrior


# Ground model
class SeismicModel(object):
    """
    Class for the seismic model.
    """

    # Constructor
    def __init__(self, model: np.array):
        """
        Constructor.

        :param model: Model parameters as (vs, z) np.array
        """
        self._model = model
    # end __init__

    @property
    def model(self) -> np.ndarray:
        """Return the full (vs, z) model array."""
        return self._model
    # end model

    @property
    def vs(self) -> np.ndarray:
        """
        Return Vs values.
        """
        n = self.nlayers
        return self._model[:n]
    # end vs

    @property
    def z(self) -> np.ndarray:
        """
        Return depth values.
        """
        n = self.nlayers
        return self._model[-n:]
    # end z

    @property
    def nlayers(self) -> int:
        """
        Return the number of layers.
        """
        return int(self._model.size / 2)
    # end nlayers

    # Parse the model
    def split_params(self):
        """
        Parse the model parameters.
        """
        model = self._model[~np.isnan(self._model)]
        n = self.nlayers  # layers
        vs = model[:n]
        z_vnoi = model[-n:]
        return n, vs, z_vnoi
    # end split_params

    def get_vp(self, vpvs=1.73, mantle=[4.3, 1.8]):
        """
        Return vp from vs, based on crustal and mantle vpvs.

        :param vpvs: Vp/Vs ratio
        :param mantle: Mantle parameters (optional)
        :return: P-wave velocity
        """
        ind_m = np.where((self.vs >= mantle[0]))[0]  # mantle
        vp = self.vs * vpvs  # correct for crust
        if len(ind_m) == 0:
            return vp
        else:
            ind_m[0] == np.int
            vp[ind_m[0]:] = self.vs[ind_m[0]:] * mantle[1]
        # end if

        return vp
    # end get_vp

    def get_vp_vs_h(self, vpvs=1.73, mantle=None):
        """
        Return vp, vs and h from a input model [vs, z_vnoi]

        :param vpvs: Vp/Vs ratio
        :param mantle: Mantle parameters (optional)
        :return: Tuple of (vp, vs, h)
        """
        # Split model parameters
        n, vs, z_vnoi = self.split_params()

        # Discontinuities
        z_disc = (z_vnoi[:n - 1] + z_vnoi[1:n]) / 2.
        h_lay = (z_disc - np.concatenate(([0], z_disc[:-1])))
        h = np.concatenate((h_lay, [0]))

        # Check for NaN values
        if mantle is not None:
            vp = self.get_vp(vpvs, mantle)
        else:
            vp = vs * vpvs
        # end if

        return vp, vs, h
    # end get_vp_vs_h

    def __str__(self):
        """
        String representation of the seismic model.
        :return: String representation
        """
        lines = ["SeismicModel:"]
        for i, (v, d) in enumerate(zip(self.vs, self.z)):
            lines.append(f"  Layer {i + 1}: Vs = {v:.3f} km/s, Depth = {d:.2f} km")
        # end for
        return "\n".join(lines)
    # end __str__

    def __repr__(self):
        """
        Representation of the seismic model.
        :return: String representation
        """
        return f"<SeismicModel with {len(self.vs)} layers>"
    # end __repr__

# end SeismicModel


# Forward modeling of dispersion curves based on surf96 (Rob Herrmann).
class SurfDispModel(object):
    """
    Use surf96 for Forward Modeling (Rob Herrmann).

    The quick fortran routine is from Hongjian Fang:
        https://github.com/caiweicaiwei/SurfTomo

    BayHunter.SurfDisp leaning on the python wrapper of Marius Isken:
        https://github.com/miili/pysurf96
    """

    def __init__(
            self,
            obsx,
            ref
    ):
        """
        Constructor

        :param obsx:
        :param ref:
        """
        # Parameters
        self.obsx = obsx
        self.kmax = obsx.size
        self.ref = ref

        # Model parameters
        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
        }

        # ...
        self.wavetype, self.veltype = self.get_surftags(ref)

        # Maximum size of period vector
        if self.kmax > KMAX_MAX:
            # Replace obsx with interpolated values
            self.obsx_int = np.linspace(obsx.min(), obsx.max(), KMAX_MAX)
            print(ERROR_KMAX_OVERRIDE_STRING)
        # end if
    # end __init__

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_surftags(self, ref):
        """
        Returns the wave type and velocity type based on the reference.

        :param ref: Reference string (e.g., 'rdispgr', 'ldispgr', etc.)
        :return:
        """
        if ref == 'rdispgr':
            return 2, 1
        elif ref == 'ldispgr':
            return 1, 1
        elif ref == 'rdispph':
            return 2, 0
        elif ref == 'ldispph':
            return 1, 0
        else:
            tagerror = "Reference is not available in SurfDisp. If you defined \
a user Target, assign the correct reference (target.ref) or update the \
forward modeling plugin with target.update_plugin(MyForwardClass()).\n \
* Your ref was: %s\nAvailable refs are: rdispgr, ldispgr, rdispph, ldispph\n \
(r=rayleigh, l=love, gr=group, ph=phase)" % ref
            raise ReferenceError(tagerror)
        # end if
    # end get_surftags

    def run(self, h, vp, vs, rho, **params):
        """
        The forward model will be run with the parameters below.

        :param h: layer thicknesses (km)
        :param vp: P-wave velocities (km/s)
        :param vs: S-wave velocities (km/s)
        :param rho: layer densities (g/cm³)
        :param params: additional parameters
        :return: tuple of (period vector, dispersion velocities)

        thkm, vpm, vsm, rhom: model for dispersion calculation
        nlayer - I4: number of layers in the model
        iflsph - I4: 0 flat earth model, 1 spherical earth model
        iwave - I4: 1 Love wave, 2 Rayleigh wave
        mode - I4: ith mode of surface wave, 1 fundamental, 2 first higher, ...
        igr - I4: 0 phase velocity, > 0 group velocity
        kmax - I4: number of periods (t) for dispersion calculation
        t - period vector (t(NP))
        cg - output phase or group velocities (vector,cg(NP))
        """
        nlayer = len(h)
        h, vp, vs, rho = SurfDispModel.get_modelvectors(h, vp, vs, rho)

        # Model parameters
        iflsph = self.modelparams['flsph']
        mode = self.modelparams['mode']
        iwave = self.wavetype
        igr = self.veltype

        # Check maximum size of period vector
        if self.kmax > KMAX_MAX:
            kmax = KMAX_MAX
            pers = self.obsx_int
        else:
            pers = np.zeros(KMAX_MAX)
            kmax = self.kmax
            pers[:kmax] = self.obsx
        # end if

        dispvel = np.zeros(KMAX_MAX)  # result

        # Call Fortran code
        error = surfdisp96(
            h,
            vp,
            vs,
            rho,
            nlayer,
            iflsph,
            iwave,
            mode,
            igr,
            kmax,
            pers,
            dispvel
        )

        # No error
        if error == 0:
            if self.kmax > KMAX_MAX:
                disp_int = np.interp(self.obsx, pers, dispvel)
                return self.obsx, disp_int
            # end if
            return pers[:kmax], dispvel[:kmax]
        # end if

        # Return NaN if error
        return np.nan, np.nan
    # end run

    @staticmethod
    def get_modelvectors(
            h,
            vp,
            vs,
            rho
    ):
        """
        Returns thkm, vpm, vsm, rhom as numpy array of size 100.

        :param h: Layer thicknesses (km)
        :param vp: P-wave velocities (km/s)
        :param vs: S-wave velocities (km/s)
        :param rho: Layer densities (g/cm³)
        :return: thkm, vpm, vsm, rhom as numpy arrays
        """
        nlayer = len(h)
        thkm = np.zeros(100)
        thkm[:nlayer] = h

        vpm = np.zeros(100)
        vpm[:nlayer] = vp

        vsm = np.zeros(100)
        vsm[:nlayer] = vs

        rhom = np.zeros(100)
        rhom[:nlayer] = rho

        return thkm, vpm, vsm, rhom
    # end get_modelvectors

# end SurfDispModel
