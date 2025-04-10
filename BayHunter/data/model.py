#
# MIGRATE
# Copyright Nils Schaetti
#

from typing import Optional, Tuple
import numpy as np
import json
import yaml
import pickle
import pyarrow as pa
import matplotlib.pyplot as plt

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
        noise_corr: Optional[float] = 0.0,
        noise_sigma: Optional[Tuple[float, float]] = (1e-5, 0.05),
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
        self.noise_corr = noise_corr
        self.noise_sigma = noise_sigma
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
            noise_corr=parse(d.get("swdnoise_corr", 0.0)),
            noise_sigma=parse(d.get("swdnoise_sigma", (1e-5, 0.05))),
        )
    # end from_dict

    def __str__(self):
        return (
            f"SeismicPrior(vs={self.vs}, z={self.z}, layers={self.layers}, "
            f"vpvs={self.vpvs}, mohoest={self.mohoest}, mantle={self.mantle}, "
            f"noise_corr={self.noise_corr}, "
            f"noise_sigma={self.noise_sigma})"
        )
    # end __str__

# end SeismicPrior


class DispersionCurve:
    """
    Class representing a dispersion curve: vg(T).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        wave_type: str = "Rayleigh",
        velocity_type: str = "group",
        source: str = "simulated"
    ):
        """
        Constructor.

        :param x: Periods (T) in seconds (1D array)
        :param y: Corresponding velocities (vg or vp) in km/s (1D array)
        :param wave_type: 'Rayleigh' or 'Love'
        :param velocity_type: 'group' or 'phase'
        :param source: 'simulated', 'observed', 'forwarded', etc.
        """
        assert len(x) == len(y), "Period and velocity arrays must match in length."
        self.x = np.array(x)
        self.y = np.array(y)
        self.wave_type = wave_type
        self.velocity_type = velocity_type
        self.source = source
    # end __init__

    # region PUBLIC

    def as_array(self) -> np.ndarray:
        """
        Returns the dispersion curve as a (2, N) numpy array (periods, velocities).
        """
        return np.stack((self.x, self.y), axis=0)
    # end as_array

    def plot(self, ax=None, label=None, color=None, linestyle='-'):
        """
        Plot the dispersion curve using matplotlib.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        # end if
        ax.plot(self.x, self.y, label=label or self.source, color=color, linestyle=linestyle)
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Group Velocity (km/s)")
        ax.set_title("Dispersion Curve")
        ax.grid(True)
        return ax
    # end plot

    # Convert to dictionary
    def to_dict(self) -> dict:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "wave_type": self.wave_type,
            "velocity_type": self.velocity_type,
            "source": self.source
        }
    # end to_dict

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        # end with
    # end save_json

    def save_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)
        # end with
    # end save_yaml

    def save_pickle(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # end with
    # end save_pickle

    def save_npz(self, path: str):
        """
        Save the dispersion curve to a .npz file.
        :param path: Path to the .npz file
        """
        np.savez(
            path,
            x=self.x,
            y=self.y,
            wave_type=self.wave_type,
            velocity_type=self.velocity_type,
            source=self.source
        )
    # end save_npz

    # endregion PUBLIC

    # region OVERRIDE

    # How many points
    def __len__(self):
        return len(self.x)
    # end __len__

    # String representation
    def __str__(self):
        return (
            f"DispersionCurve[{self.source}]: {len(self.x)} points | "
            f"{self.wave_type}-{self.velocity_type} wave"
        )
    # end __str__

    # Representation
    def __repr__(self):
        return f"<DispersionCurve len={len(self)} type={self.wave_type}-{self.velocity_type}>"
    # end __repr__

    # endregion OVERRIDE

    # region CLASSMETHODS

    @classmethod
    def from_dict(cls, data: dict) -> "DispersionCurve":
        return cls(
            x=np.array(data["x"]),
            y=np.array(data["y"]),
            wave_type=data.get("wave_type", "Rayleigh"),
            velocity_type=data.get("velocity_type", "group"),
            source=data.get("source", "simulated")
        )
    # end from_dict

    @classmethod
    def load_json(cls, path: str) -> "DispersionCurve":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
        # end with
    # end load_json

    @classmethod
    def load_yaml(cls, path: str) -> "DispersionCurve":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
        # end with
    # end load_yaml

    @classmethod
    def load_pickle(cls, path: str) -> "DispersionCurve":
        with open(path, "rb") as f:
            return pickle.load(f)
        # end with
    # end load_pickle

    @classmethod
    def load_npz(cls, path: str) -> "DispersionCurve":
        """
        Load a dispersion curve from a .npz file.
        :param path: Path to the .npz file
        :return: DispersionCurve object
        """
        data = np.load(path, allow_pickle=True)
        return cls(
            x=data["x"],
            y=data["y"],
            wave_type=str(data["wave_type"]),
            velocity_type=str(data["velocity_type"]),
            source=str(data["source"])
        )
    # end load_npz

    # endregion CLASSMETHODS

# end DispersionCurve



# Ground model
class SeismicModel(object):
    """
    Class for the seismic model.
    """

    # Constructor
    def __init__(
            self,
            model: np.array,
            vpvs: float
    ):
        """
        Constructor.

        :param model: Model parameters as (vs, z) np.array
        :param vpvs: Vp/Vs ratio
        """
        self._model = model
        self._vpvs = vpvs
    # end __init__

    # region PROPERTIES

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
    def vpvs(self) -> float:
        """
        Return Vp/Vs ratio.
        """
        return self._vpvs
    # end vpvs

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

    # endregion PROPERTIES

    # region PUBLIC

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

    def get_vp(self, mantle=[4.3, 1.8]):
        """
        Return vp from vs, based on crustal and mantle vpvs.

        :param vpvs: Vp/Vs ratio
        :param mantle: Mantle parameters (optional)
        :return: P-wave velocity
        """
        ind_m = np.where((self.vs >= mantle[0]))[0]  # mantle
        vp = self.vs * self.vpvs  # correct for crust
        if len(ind_m) == 0:
            return vp
        else:
            ind_m[0] == np.int
            vp[ind_m[0]:] = self.vs[ind_m[0]:] * mantle[1]
        # end if

        return vp
    # end get_vp

    def get_vp_vs_h(self, mantle=None):
        """
        Return vp, vs and h from a input model [vs, z_vnoi]

        :param mantle: Mantle parameters (optional)
        :return: Tuple of (vp, vs, h)
        """
        # Split model parameters
        n, vs, z_vnoi = self.split_params()

        # Middle points of discontinuities
        z_disc = (z_vnoi[:n - 1] + z_vnoi[1:n]) / 2.

        # Relative layer thicknesses (middle points relative to the previous)
        h_lay = (z_disc - np.concatenate(([0], z_disc[:-1])))

        # Add zero thickness for the last layer (infinite half-space)
        h = np.concatenate((h_lay, [0]))

        # Check for NaN values
        if mantle is not None:
            vp = self.get_vp(self.vpvs, mantle)
        else:
            vp = vs * self.vpvs
        # end if

        return vp, vs, h
    # end get_vp_vs_h

    # Run model
    def forward(
            self,
            length: int = 60,
    ):
        """
        Run the forward model.

        :param length: Length of the dispersion curve.
        :return: DispersionCurve object
        """
        # Get model parameters
        vp, vs, h = self.get_vp_vs_h()

        # Compute layer densities from Vp
        rho = vs * 1.73

        # Compute synthetic dispersion curves
        x, y = self._calc_synth(
            length=length,
            h=h,
            vp=vp,
            vs=vs,
            rho=rho
        )

        # Create DispersionCurve object
        return DispersionCurve(
            x=x,
            y=y,
            wave_type="Rayleigh",
            velocity_type="group",
            source="synthetic"
        )
    # end forward

    def plot(self, ax=None, title="Model", invert_axes=False, label=None, color=None, linestyle='-'):
        """
        Plot the seismic model.

        :param ax: Matplotlib Axes object (optional)
        :param title: Title of the plot
        :param invert_axes: If True, plot velocity on X and depth on Y
        :param label: Optional legend label
        :param color: Optional curve color
        :param linestyle: Line style (default: '-')
        :return: Axes object
        """
        vp, vs, h = self.get_vp_vs_h()
        z_interfaces = np.cumsum(h)
        z_interfaces[-1] = self.z[-1]  # assurer cohérence avec profondeur finale

        # Étend la courbe en escalier
        vs_plot = np.repeat(vs, 2)
        z_plot = np.zeros_like(vs_plot)
        z_plot[1:-1:2] = z_interfaces[:-1]
        z_plot[2::2] = z_interfaces[:-1]
        z_plot[-1] = z_interfaces[-1]

        if ax is None:
            figsize = (6, 10) if invert_axes else (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
        # end if

        if invert_axes:
            ax.plot(vs_plot, z_plot, label=label, color=color, linestyle=linestyle)
            ax.set_xlabel("Velocity (km/s)")
            ax.set_ylabel("Depth (km)")
            ax.invert_yaxis()
        else:
            ax.plot(z_plot, vs_plot, label=label, color=color, linestyle=linestyle)
            ax.set_xlabel("Depth (km)")
            ax.set_ylabel("Velocity (km/s)")
        # end if

        ax.set_title(title)
        ax.grid(True)
        return ax
    # end plot

    # Object to dictionary
    def to_dict(self) -> dict:
        """
        Convert the model to a dictionary.
        :return: Dictionary representation of the model
        """
        return {
            "vs": self.vs.tolist(),
            "z": self.z.tolist(),
            "vpvs": self.vpvs
        }
    # end to_dict

    # Save model to JSON
    def save_json(self, path: str):
        """
        Save the model to a JSON file.
        :param path: Path to the JSON file
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        # end with
    # end save_json

    # Save model to YAML
    def save_yaml(self, path: str):
        """
        Save the model to a YAML file.
        :param path: Path to the YAML file
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)
        # end with
    # end save_yaml

    def save_pickle(self, path: str):
        """
        Save the model to a pickle file.
        :param path: Path to the pickle file
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # end with
    # end save_pickle

    # endregion PUBLIC

    # region PRIVATE

    def _calc_synth(
            self,
            length: int,
            h: np.ndarray,
            vp: np.ndarray,
            vs: np.ndarray,
            rho: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate synthetic dispersion curves.

        :param length: Length of the dispersion curve.
        :param h: Layer thicknesses (km)
        :param vp: P-wave velocities (km/s)
        :param vs: S-wave velocities (km/s)
        :param rho: Layer densities (g/cm³)
        :return: Tuple of (period vector, dispersion velocities)
        """
        # Create plugin object
        plugin = SurfDispModel(
            kmax=length,
            min_p=1.0,
            max_p=15.0
        )

        # Call Fortran code
        return plugin.run(
            h,
            vp,
            vs,
            rho
        )
    # end _calc_synth

    # endregion PRIVATE

    # region OVERRIDE

    def __str__(self):
        """
        String representation of the seismic model.
        :return: String representation
        """
        # Get model parameters
        vp, vs, h = self.get_vp_vs_h()

        # Header
        lines = ["SeismicModel:"]
        lines += [f" Number of layers: {self.nlayers}"]
        lines += [f"  Vp/Vs ratio: {self.vpvs:.2f}"]

        # Layers
        for i, (v, d) in enumerate(zip(self.vs, self.z)):
            lines.append(
                f"  Layer {i + 1}: Vs = {v:.3f} km/s, Vp = {vp[i]:.3f} km/s, "
                f"Depth = {d:.2f} km, Thickness = {h[i]:.2f} km"
            )
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

    # endregion OVERRIDE

    # region CLASSMETHODS

    # Load model from JSON file
    @classmethod
    def from_dict(cls, data: dict) -> "SeismicModel":
        """
        Create a SeismicModel from a dictionary.
        :param data: Dictionary containing model parameters.
        :return: SeismicModel object.
        """
        vs = np.array(data["vs"])
        z = np.array(data["z"])
        model = np.concatenate([vs, z])
        return cls(model=model, vpvs=data["vpvs"])
    # end from_dict

    # Load model from YAML file
    @classmethod
    def load_json(cls, path: str) -> "SeismicModel":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
        # end with
    # end load_json

    # Load model from YAML file
    @classmethod
    def load_yaml(cls, path: str) -> "SeismicModel":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
        # end with
    # end load_yaml

    # Load model from Pickle file
    @classmethod
    def load_pickle(cls, path: str) -> "SeismicModel":
        with open(path, "rb") as f:
            return pickle.load(f)
        # end with
    # end load_pickle

    # endregion CLASSMETHODS

# end SeismicModel


class SeismicSample:
    """
    Class representing a seismic sample, including the model and dispersion curve.
    """

    # Constructor
    def __init__(self, model: SeismicModel, curve: DispersionCurve):
        self.model = model
        self.curve = curve
    # end __init__

    # To dictionary
    def to_arrow_dict(self) -> dict:
        return {
            "vs": self.model.vs.astype(np.float32).tolist(),
            "z": self.model.z.astype(np.float32).tolist(),
            "vpvs": float(self.model.vpvs),
            "disp_x": self.curve.x.astype(np.float32).tolist(),
            "disp_y": self.curve.y.astype(np.float32).tolist(),
            "wave_type": self.curve.wave_type,
            "velocity_type": self.curve.velocity_type,
        }
    # end to_arrow_dict

# end SeismicSample


# Batch of seismic samples
class SeismicSampleBatch:
    """
    Class representing a batch of seismic samples.
    """

    def __init__(self):
        """
        Constructor for SeismicSampleBatch.
        """
        self.samples = []
    # end __init__

    def add(self, sample: SeismicSample):
        """
        Add a seismic sample to the batch.
        :param sample: SeismicSample object to add
        """
        self.samples.append(sample)
    # end add

    def to_arrow_table(self) -> pa.Table:
        """
        Convert the batch of seismic samples to an Arrow table.
        :return: Arrow table representation of the batch
        """
        dicts = [s.to_arrow_dict() for s in self.samples]
        return pa.table({key: [d[key] for d in dicts] for key in dicts[0].keys()})
    # end to_arrow_table

# end SeismicSampleBatch


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
            kmax: int,
            min_p: float,
            max_p: float
    ):
        """
        Constructor

        :param kmax: Maximum number of periods
        :param min_p: Minimum period
        :param max_p: Maximum period
        """
        # Parameters
        self._kmax = kmax
        self._min_p = min_p
        self._max_p = max_p

        # Model parameters
        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
        }

        # Wavetype and velocity type
        self.wavetype, self.veltype = 2, 1

        # Maximum size of period vector
        self._obsx = np.linspace(min_p, max_p, kmax)
        if self._kmax > KMAX_MAX:
            # Replace obsx with interpolated values
            self._obsx_int = np.linspace(min_p, max_p, KMAX_MAX)
            print(ERROR_KMAX_OVERRIDE_STRING)
        # end if
    # end __init__

    # region PROPERTIES

    @property
    def kmax(self) -> int:
        """
        Return the maximum number of periods.
        """
        return self._kmax
    # end kmax

    @property
    def min_p(self) -> float:
        """
        Return the minimum period.
        """
        return self._min_p
    # end min_p

    @property
    def max_p(self) -> float:
        """
        Return the maximum period.
        """
        return self._max_p
    # end max_p

    # endregion PROPERTIES

    # region PUBLIC

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)
    # end set_modelparams

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
            pers = self._obsx_int
        else:
            pers = np.zeros(KMAX_MAX)
            kmax = self.kmax
            pers[:kmax] = self._obsx
        # end if

        # Output value
        dispvel = np.zeros(KMAX_MAX)

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
                disp_int = np.interp(self._obsx, pers, dispvel)
                return self._obsx, disp_int
            # end if
            return pers[:kmax], dispvel[:kmax]
        # end if

        # Return NaN if error
        return np.nan, np.nan
    # end run

    # endregion PUBLIC

    # region STATIC

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

    # endregion STATIC

# end SurfDispModel
