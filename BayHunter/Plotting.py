# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import os.path as op
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections import OrderedDict

from BayHunter import utils, MCMC_Optimizer
from BayHunter import Targets
from BayHunter import Model, ModelMatrix
import matplotlib.colors as colors

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rstate = np.random.RandomState(333)


def vs_round(vs):
    """Round Vs values down to the nearest 0.025 km/s increment.

    Args:
        vs (float or numpy.ndarray): Shear-wave velocity values to round.

    Returns:
        numpy.ndarray: Rounded shear-wave velocities.
    """
    # rounding down to next smaller 0.025 interval
    vs_floor = np.floor(vs)
    return np.round((vs-vs_floor)*40)/40 + vs_floor


# end def vs_round
def tryexcept(func):
    """Wrap a plotting function with a broad try/except handler.

    Args:
        func (Callable): Plotting callable to wrap.

    Returns:
        Callable: Wrapped function that prints an informative message when an
            exception is raised and returns ``None`` instead of propagating the
            error.
    """
    def wrapper_tryexcept(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            print('* %s: Plotting was not possible\nErrorMessage: %s'
                  % (func.__name__, e))
            return None
    # end def wrapper_tryexcept
    return wrapper_tryexcept
# end def tryexcept


class PlotFromChains(object):
    """Plot posterior samples from terminated chains."""

    def __init__(
            self,
            initparams: dict,
            optimizer: MCMC_Optimizer
    ):
        """
        Plot previously generated chain outputs stored in the optimizer.
        """
        self.initparams = initparams
        self.save_path = initparams['savepath']
        self.n_chains = initparams['nchains']
        self.n_simulations = initparams['iter_burnin'] + initparams['iter_main']
        self.optimizer = optimizer
        self.chains_misfits = dict()
        self.simulation_counts = dict()
        self.plot_dir = Path(self.save_path) / 'plots'
        self.data_dir = Path(self.save_path) / 'data'
        self._load_chain_data()
        self._create_directories()
    # end def __init__

    def _create_directories(self):
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    # end def _create_directories

    def _load_chain_data(self):
        """
        Load chain data from the optimizer.
        """
        for c_n, c in enumerate(self.optimizer.chains):
            self.chains_misfits[c_n] = c.misfits
            self.simulation_counts[c_n] = c.simulation_counts
        # end for
    # end def _load_chain_data

    def save_plots(self):
        # Misfits per simulations plot
        self._plot_misfit_per_simulation()

        # Misfits distribution per simulations plot
        self._plot_misfit_distribution_per_simulation()
    # end def save_plots

    def _plot_misfit_per_simulation(self):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        plot_data = []
        for c_n in range(self.n_chains):
            counts = self.simulation_counts[c_n]
            misfits = self.chains_misfits[c_n]
            ax.plot(
                counts,
                misfits,
                label=f'Chain {c_n}'
            )
            plot_data.append({
                'chain': c_n,
                'simulation_counts': counts,
                'misfits': misfits
            })
        # end for
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Simulation')
        ax.set_ylabel('Misfit')
        ax.set_title('Misfits per simulation / chains')
        plt.grid(which='major', linestyle = '--', linewidth = 0.5)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        fig.tight_layout()
        fig.savefig(f'{self.plot_dir}/misfits_per_simulation.png')
        np.save(
            f'{self.data_dir}/misfits_per_simulation.npy',
            np.array(plot_data, dtype=object)
        )
    # end def _plot_misfit_per_simulation

    def _plot_misfit_distribution_per_simulation(self):
        """
        Plot the distribution of misfits per simulation.
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        data_x = list()
        data_mean = list()
        data_median = list()
        data_q1 = list()
        data_q3 = list()
        data_95 = list()
        data_05 = list()
        data_min = list()
        data_max = list()

        step = 10
        best_per_chain = int(step / self.n_chains)
        for n_sim in range(step, self.n_simulations, step):
            total_sim = 0
            step_best_misfits = list()
            for c_n in range(self.n_chains):
                c_misfits = self.chains_misfits[c_n][self.simulation_counts[c_n] <= n_sim]
                step_best_misfits.append(c_misfits[-best_per_chain:].reshape(1, -1))
                c_sims = self.simulation_counts[c_n][self.simulation_counts[c_n] <= n_sim]
                total_sim += c_sims[-1]
            # end for
            step_misfits = np.concatenate(step_best_misfits, axis=0)
            data_x.append(total_sim)
            data_mean.append(np.mean(step_misfits))
            data_median.append(np.median(step_misfits))
            data_q1.append(np.quantile(step_misfits, 0.25))
            data_q3.append(np.quantile(step_misfits, 0.75))
            data_95.append(np.quantile(step_misfits, 0.95))
            data_05.append(np.quantile(step_misfits, 0.05))
            data_min.append(np.min(step_misfits))
            data_max.append(np.max(step_misfits))
        # end for

        data_x = np.array(data_x)
        data_mean = np.array(data_mean)
        data_median = np.array(data_median)
        data_q1 = np.array(data_q1)
        data_q3 = np.array(data_q3)
        data_95 = np.array(data_95)
        data_05 = np.array(data_05)
        data_min = np.array(data_min)
        data_max = np.array(data_max)

        ax.plot(data_x, data_mean, c='b', label="MCMC mean")
        ax.plot(data_x, data_median, c='r', label="MCMC median")
        ax.plot(data_x, data_q1, c='orange', label="MCMC Q1")
        ax.plot(data_x, data_q3, c='orange', label="MCMC Q3")
        ax.fill_between(data_x, data_q1, data_q3, alpha=0.2, color='blue')
        ax.plot(data_x, data_min, c='blue', linestyle='--', label="MCMC 95%")
        ax.plot(data_x, data_max, c='blue', linestyle='--', label="MCMC 5%")
        ax.set_xlabel('Total simulations')
        ax.set_ylabel('Misfit (RMS)')
        ax.legend()

        ax.set_title('Misfits distribution per simulation')

        fig.tight_layout()

        plt.grid(which='major', linestyle='--', linewidth=0.5)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)

        # Save fig
        fig.savefig(f"{self.plot_dir}/misfits_distribution_per_simulation.png")

        # Save data
        np.save(
            f"{self.data_dir}/misfits_distribution_per_simulation.npy",
            np.array([data_x, data_mean, data_median, data_q1, data_q3, data_95, data_05, data_min, data_max])
        )
    # end def _plot_misfit_distribution_per_simulation

# end class PlotFromChains


class PlotFromStorage(object):
    """Plot previously generated chain outputs stored on disk.

    The helper reads the inversion configuration, loads accepted chain values,
    and offers a catalogue of plotting utilities without requiring a running
    ``SingleChain`` instance.

    Attributes:
        targets (Targets): Target collection loaded from the configuration.
        priors (dict): Prior bounds applied in the inversion.
        initparams (dict): Inversion control parameters.
        datapath (str): Directory containing ``c???_p?.npy`` files to plot.
        figpath (str): Output directory for generated figures.
    """
    def __init__(
            self,
            configfile
    ):
        """Load configuration metadata and locate stored chain artifacts.

        Args:
            configfile (str): Path to a configuration file that references the
                ``c???_p?.npy`` files to visualize.
        """
        condict = self.read_config(configfile)
        self.targets = condict['targets']
        self.ntargets = len(self.targets)
        self.refs = condict['targetrefs'] + ['joint']
        self.priors = condict['priors']
        self.initparams = condict['initparams']

        self.datapath = op.dirname(configfile)
        self.figpath = str(Path(self.datapath).parent / "plots")

        self.init_filelists()
        self.init_outlierlist()

        self.mantle = self.priors.get('mantle', None)

        self.refmodel = {
            'model': None,
            'nlays': None,
            'noise': None,
            'vpvs': None
        }
    # end def __init__

    def read_config(self, configfile):
        """Read the BayHunter configuration file.

        Args:
            configfile (str): Path to the configuration file.

        Returns:
            dict: Parsed configuration dictionary matching
            :func:`BayHunter.utils.read_config`.
        """
        return utils.read_config(configfile)
    # end def read_config

    def savefig(self, fig, filename):
        """Persist a figure if it is not ``None``.

        Args:
            fig (matplotlib.figure.Figure or None): Figure to save.
            filename (str): File name relative to ``self.figpath``.
        """
        if fig is not None:
            outfile = op.join(self.figpath, filename)
            fig.savefig(outfile, bbox_inches="tight")
            plt.close('all')
        # end if
    # end def savefig

    def init_outlierlist(self):
        """Load outlier chain indices from disk if they exist.

        Returns:
            None
        """
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            self.outliers = np.loadtxt(outlierfile, usecols=[0], dtype=int)
            print('Outlier chains from file: %d' % self.outliers.size)
        else:
            print('Outlier chains from file: None')
            self.outliers = np.zeros(0)
        # end if
    # end def init_outlierlist

    def init_filelists(self):
        """Collect phase-specific file lists for every stored data type.

        Returns:
            None
        """
        filetypes = ['models', 'likes', 'misfits', 'noise', 'vpvs']
        filepattern = op.join(self.datapath, 'c???_p%d%s.npy')
        files = []
        size = []

        for ftype in filetypes:
            p1files = sorted(glob.glob(filepattern % (1, ftype)))
            p2files = sorted(glob.glob(filepattern % (2, ftype)))
            files.append([p1files, p2files])
            size.append(len(p1files) + len(p2files))
        # end for

        if len(set(size)) == 1:
            self.modfiles, self.likefiles, self.misfiles, self.noisefiles, \
                self.vpvsfiles = files
        else:
            logger.info(
                'You are missing files. Please check ' + '"%s" for completeness.' % self.datapath
            )
            logger.info('(filetype, number): ' + str(zip(filetypes, size)))
        # end if
    # end def init_filelists

    def get_outliers(self, dev):
        """Determine chains with underperforming likelihoods.

        Args:
            dev (float): Acceptable fractional deviation (0-1) from the best
                median likelihood before a chain is flagged as an outlier.

        Returns:
            numpy.ndarray: Chain indices marked as outliers according to the
            deviation criterion.
        """
        nchains = len(self.likefiles[1])
        chainidxs = np.zeros(nchains) * np.nan
        chainmedians = np.zeros(nchains) * np.nan

        for i, likefile in enumerate(self.likefiles[1]):
            cidx, _, _ = self._return_c_p_t(likefile)
            chainlikes = np.load(likefile)
            chainmedian = np.median(chainlikes)

            chainidxs[i] = cidx
            chainmedians[i] = chainmedian
        # end for

        maxlike = np.max(chainmedians)  # best chain average

        # scores must be smaller 1
        if maxlike > 0:
            scores = chainmedians / maxlike
        elif maxlike < 0:
            scores = maxlike / chainmedians
        # end if

        # end if
        outliers = chainidxs[np.where(((1-scores) > dev))]
        outscores = 1 - scores[np.where(((1-scores) > dev))]

        if len(outliers) > 0:
            print('Outlier chains found with following chainindices:\n')
            print(outliers)
            outlierfile = op.join(self.datapath, 'outliers.dat')
            with open(outlierfile, 'w') as f:
                f.write('# Outlier chainindices with %.3f deviation condition\n' % dev)
                for i, outlier in enumerate(outliers):
                    f.write('%d\t%.3f\n' % (outlier, outscores[i]))

                # end for
            # end with
        # end if

        return outliers
    # end def get_outliers

    def _get_chaininfo(self):
        """Return chain indices and model counts for posterior phase files.

        Returns:
            tuple: ``(chain_indices, nmodels)`` lists aligned with phase-2
            likelihood files.
        """
        nmodels = [len(np.load(file)) for file in self.likefiles[1]]
        chainlist = [self._return_c_p_t(file)[0] for file in self.likefiles[1]]
        return chainlist, nmodels

    # end def _get_chaininfo
    def save_final_distribution(self, maxmodels=200000, dev=0.05):
        """Combine the posterior results from all non-outlier chains.

        Args:
            maxmodels (int, optional): Upper bound for the final number of
                stored posterior samples. Defaults to ``200000``.
            dev (float, optional): Fractional likelihood drop tolerated before
                rejecting a chain as an outlier. Defaults to ``0.05``.

        Returns:
            None: The combined posterior arrays ``c_models``, ``c_likes``,
            ``c_misfits``, ``c_noise``, and ``c_vpvs`` are saved directly to
            ``self.datapath``.
        """

        def save_finalmodels(models, likes, misfits, noise, vpvs):
            """Write the combined posterior arrays to ``self.datapath``."""
            names = ['models', 'likes', 'misfits', 'noise', 'vpvs']
            print('> Saving posterior distribution.')
            for i, data in enumerate([models, likes, misfits, noise, vpvs]):
                outfile = op.join(self.datapath, 'c_%s' % names[i])
                np.save(outfile, data)

        # delete old outlier file if evaluating outliers newly
            # end for
        # end def save_finalmodels
        outlierfile = op.join(self.datapath, 'outliers.dat')
        if op.exists(outlierfile):
            os.remove(outlierfile)

        # end if
        self.outliers = self.get_outliers(dev=dev)

        # due to the forced acceptance rate, each chain should have accepted
        # a similar amount of models. Therefore, a constant number of models
        # will be considered from each chain (excluding outlier chains), to
        # add up to a collection of maxmodels models.
        nchains = int(len(self.likefiles[1]) - self.outliers.size)
        maxmodels = int(maxmodels)
        mpc = int(maxmodels / nchains)  # models per chain

        # # open matrixes and vectors
        allmisfits = None
        allmodels = None
        alllikes = np.ones(maxmodels) * np.nan
        allnoise = np.ones((maxmodels, self.ntargets*2)) * np.nan
        allvpvs = np.ones(maxmodels) * np.nan

        start = 0
        chainidxs, nmodels = self._get_chaininfo()

        for i, cidx in enumerate(chainidxs):
            if cidx in self.outliers:
                continue

            # end if
            index = np.arange(nmodels[i]).astype(int)
            if nmodels[i] > mpc:
                index = rstate.choice(index, mpc, replace=False)
                index.sort()

            # end if
            chainfiles = [self.modfiles[1][i], self.misfiles[1][i],
                          self.likefiles[1][i], self.noisefiles[1][i],
                          self.vpvsfiles[1][i]]

            for c, chainfile in enumerate(chainfiles):
                _, _, ftype = self._return_c_p_t(chainfile)
                data = np.load(chainfile)[index]

                if c == 0:
                    end = start + len(data)

                # end if
                if ftype == 'likes':
                    alllikes[start:end] = data

                elif ftype == 'models':
                    if allmodels is None:
                        allmodels = np.ones((maxmodels, data[0].size)) * np.nan

                    # end if
                    allmodels[start:end, :] = data

                elif ftype == 'misfits':
                    if allmisfits is None:
                        allmisfits = np.ones((maxmodels, data[0].size)) * np.nan

                    # end if
                    allmisfits[start:end, :] = data

                elif ftype == 'noise':
                    allnoise[start:end, :] = data

                elif ftype == 'vpvs':
                    allvpvs[start:end] = data

                # end if
            # end for
            start = end

        # end for
        allmodels = allmodels[~np.isnan(alllikes)]
        allmisfits = allmisfits[~np.isnan(alllikes)]
        allnoise = allnoise[~np.isnan(alllikes)]
        allvpvs = allvpvs[~np.isnan(alllikes)]
        alllikes = alllikes[~np.isnan(alllikes)]

        save_finalmodels(allmodels, alllikes, allmisfits, allnoise, allvpvs)
        
    # end def save_final_distribution
    def _unique_legend(self, handles, labels):
        """Return the latest handle for each legend label.

        Args:
            handles (list): Matplotlib handles.
            labels (list): Corresponding legend labels.

        Returns:
            tuple: Unique handles and labels preserved as insertion-ordered
            lists.
        """
        legend = OrderedDict(zip(labels, handles))
        return legend.values(), legend.keys()

    # end def _unique_legend
    def _return_c_p_t(self, filename):
        """Extract chain metadata from a stored filename.

        Args:
            filename (str): File path following ``c???_p?.npy`` naming.

        Returns:
            tuple: ``(chainidx, phase, ftype)`` where ``phase`` is ``p1`` or
            ``p2`` and ``ftype`` is the trailing token (e.g., ``models``).
        """
        c, pt = op.basename(filename).split('.npy')[0].split('_')
        cidx = int(c[1:])
        phase, ftype = pt[:2], pt[2:]

        return cidx, phase, ftype

    # end def _return_c_p_t
    def _sort(self, chainidxstring):
        """Convert ``c???`` identifiers into sortable integers.

        Args:
            chainidxstring (str): Chain identifier string (e.g. ``c001``).

        Returns:
            int: Integer chain index.
        """
        chainidx = int(chainidxstring[1:])
        return chainidx

    # end def _sort
    def _get_layers(self, models):
        """Count the number of layers in each model array.

        Args:
            models (numpy.ndarray): Collection of stacked model vectors.

        Returns:
            numpy.ndarray: Layer counts per model.
        """
        layernumber = np.array([(len(model[~np.isnan(model)]) / 2 - 1)
                                for model in models])
        return layernumber

    # end def _get_layers
    @tryexcept
    def plot_refmodel(self, fig, mtype='model', **kwargs):
        """Overlay stored reference values onto an existing figure.

        Args:
            fig (matplotlib.figure.Figure): Figure returned by other plotting
                helpers.
            mtype (str, optional): Reference type to plot. One of
                ``model``, ``nlays``, ``noise``, or ``vpvs``. Defaults to
                ``model``.
            **kwargs: Passed to the matplotlib plotting call.

        Returns:
            matplotlib.figure.Figure: The modified input figure.
        """
        if fig is not None and self.refmodel[mtype] is not None:
            if mtype == 'nlays':
                nlays = self.refmodel[mtype]
                fig.axes[0].axvline(nlays, color='red', lw=0.5, alpha=0.7)

            # end if
            if mtype == 'model':
                dep, vs = self.refmodel['model']
                assert len(dep) == len(vs)
                fig.axes[0].plot(vs, dep, **kwargs)
                if len(fig.axes) == 2:
                    deps = np.unique(dep)
                    for d in deps:
                        fig.axes[1].axhline(d, **kwargs)

                    # end for
                # end if
            # end if
            if mtype == 'noise':
                noise = self.refmodel[mtype]
                for i in range(len(noise)):
                    fig.axes[i].axvline(
                        noise[i], color='red', lw=0.5, alpha=0.7)

                # end for
            # end if
            if mtype == 'vpvs':
                vpvs = self.refmodel[mtype]
                fig.axes[0].axvline(vpvs, color='red', lw=0.5, alpha=0.7)
            # end if
        # end if
        return fig

    # Plot values per iteration.

    # end def plot_refmodel
    def _plot_iitervalues(self, files, ax, layer=0, misfit=0, noise=0, ind=-1):
        """Draw iteration-wise traces for models, misfits, noise, or vp/vs.

        Args:
            files (list[str]): File paths containing per-iteration values.
            ax (matplotlib.axes.Axes): Axis to draw on.
            layer (bool, optional): Plot the number of layers instead of raw
                values. Defaults to ``0``.
            misfit (bool, optional): Plot misfit series. Defaults to ``0``.
            noise (bool, optional): Plot noise parameters. Defaults to ``0``.
            ind (int, optional): Component index used when ``misfit`` or
                ``noise`` is selected. Defaults to ``-1``.

        Returns:
            matplotlib.axes.Axes: Axis containing the plotted lines.
        """
        unifiles = set([f.replace('p1', 'p2') for f in files])
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, len(unifiles)))

        xmin = -self.initparams['iter_burnin']
        xmax = self.initparams['iter_main']

        files.sort()
        n = 0
        for i, file in enumerate(files):
            phase = int(op.basename(file).split('_p')[1][0])
            alpha = (0.4 if phase == 1 else 0.7)
            ls = ('-' if phase == 1 else '-')
            lw = (0.5 if phase == 1 else 0.8)
            chainidx, _, _ = self._return_c_p_t(file)
            color = color_list[n]

            data = np.load(file)
            if layer:
                data = self._get_layers(data)
            # end if
            if misfit or noise:
                data = data.T[ind]

            # end if
            iters = (np.linspace(xmin, 0, data.size) if phase == 1 else
                     np.linspace(0, xmax, data.size))
            label = 'c%d' % (chainidx)

            ax.plot(iters, data, color=color,
                    ls=ls, lw=lw, alpha=alpha,
                    label=label if phase == 2 else '')

            if phase == 2:
                if n == 0:
                    datamax = data.max()
                    datamin = data.min()
                else:
                    datamax = np.max([datamax, data.max()])
                    datamin = np.min([datamin, data.min()])
                # end if
                n += 1

            # end if
        # end for
        ax.set_xlim(xmin, xmax)
        # ax.set_ylim(datamin*0.95, datamax*1.05)
        ax.set_ylim(bottom=0)
        ax.axvline(0, color='k', ls=':', alpha=0.7)

        (abs(xmin) + xmax)
        center = np.array([abs(xmin/2.), abs(xmin) + xmax/2.]) / (abs(xmin) + xmax)
        for i, text in enumerate(['Burn-in phase', 'Exploration phase']):
            ax.text(center[i], 0.97, text,
                    fontsize=12, color='k',
                    horizontalalignment='center',
                    verticalalignment='top',
                    transform=ax.transAxes)

        # end for
        ax.set_xlabel('# Iteration')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    # end def _plot_iitervalues
    @tryexcept
    def plot_iitermisfits(self, nchains=6, ind=-1):
        """Plot misfit trajectories for a subset of chains.

        Args:
            nchains (int, optional): Number of chains to show. Defaults to 6.
            ind (int, optional): Misfit index mapping to ``self.refs``.
                Defaults to ``-1`` (joint misfit).

        Returns:
            matplotlib.figure.Figure: Misfit figure or ``None`` if plotting
            fails.
        """
        print(f"plotting {nchains} chains with misfit")
        files = self.misfiles[0][:nchains] + self.misfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(14, 8))
        ax = self._plot_iitervalues(files, ax, misfit=True, ind=ind)
        ax.set_ylabel('%s misfit' % self.refs[ind])
        return fig

    # end def plot_iitermisfits
    @tryexcept
    def plot_iiterlikes(self, nchains=6):
        """Plot likelihood trajectories for a subset of chains.

        Args:
            nchains (int, optional): Number of chains to show. Defaults to 6.

        Returns:
            matplotlib.figure.Figure: Likelihood figure or ``None`` if plotting
            fails.
        """
        files = self.likefiles[0][:nchains] + self.likefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(15, 8))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Likelihood')
        return fig

    # end def plot_iiterlikes
    @tryexcept
    def plot_iiternoise(self, nchains=6, ind=-1):
        """Plot inferred noise hyper-parameters per iteration.

        Args:
            nchains (int, optional): Number of chains to show. Defaults to 6.
            ind (int, optional): Flattened noise index. Even entries mark the
                correlation term; odd entries represent sigma. Defaults to -1.

        Returns:
            matplotlib.figure.Figure: Noise-parameter figure or ``None`` if
            plotting fails.
        """
        files = self.noisefiles[0][:nchains] + self.noisefiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(15, 8))
        ax = self._plot_iitervalues(files, ax, noise=True, ind=ind)

        parameter = np.concatenate(
            [['correlation (%s)' % ref, '$\sigma$ (%s)' % ref] for ref in self.refs[:-1]])
        ax.set_ylabel(parameter[ind])
        return fig

    # end def plot_iiternoise
    @tryexcept
    def plot_iiternlayers(self, nchains=6):
        """Plot the accepted layer count per iteration for several chains.

        Args:
            nchains (int, optional): Number of chains to show. Defaults to 6.

        Returns:
            matplotlib.figure.Figure: Layer-count figure or ``None`` if plotting
            fails.
        """
        files = self.modfiles[0][:nchains] + self.modfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(15, 8))
        ax = self._plot_iitervalues(files, ax, layer=True)
        ax.set_ylabel('Number of layers')
        return fig

    # end def plot_iiternlayers
    @tryexcept
    def plot_iitervpvs(self, nchains=6):
        """Plot vp/vs ratios per iteration for several chains.

        Args:
            nchains (int, optional): Number of chains to show. Defaults to 6.

        Returns:
            matplotlib.figure.Figure: Vp/Vs figure or ``None`` if plotting
            fails.
        """
        files = self.vpvsfiles[0][:nchains] + self.vpvsfiles[1][:nchains]

        fig, ax = plt.subplots(figsize=(15, 8))
        ax = self._plot_iitervalues(files, ax)
        ax.set_ylabel('Vp / Vs')
        return fig


# Posterior distributions as 1D histograms for noise and misfits.
# And as 2D histograms / 1D plot for final velocity-depth models
# Considering weighted models.

    # end def plot_iitervpvs
    @staticmethod
    def _plot_bestmodels(bestmodels, dep_int=None):
        """Plot mean/median/std models as line plots.

        Args:
            bestmodels (numpy.ndarray): Ensemble of models in vector form.
            dep_int (numpy.ndarray, optional): Depth grid for interpolation.

        Returns:
            tuple: ``(fig, ax)`` with the rendered matplotlib objects.
        """
        fig, ax = plt.subplots(figsize=(8, 15))

        models = ['mean', 'median', 'stdminmax']
        colors = ['green', 'blue', 'black']
        ls = ['-', '--', ':']
        lw = [1, 1, 1]

        singlemodels = ModelMatrix.get_singlemodels(bestmodels, dep_int)

        for i, model in enumerate(models):
            vs, dep = singlemodels[model]

            ax.plot(vs.T, dep, color=colors[i], label=model,
                    ls=ls[i], lw=lw[i])

        # end for
        ax.invert_yaxis()
        ax.set_ylabel('Depth in km')
        ax.set_xlabel('$V_S$ in km/s')

        han, lab = ax.get_legend_handles_labels()
        ax.legend(han[:-1], lab[:-1], loc=3)
        return fig, ax

    # end def _plot_bestmodels
    @staticmethod
    def _plot_bestmodels_hist(models, dep_int=None):
        """Visualize posterior velocities as a 2D histogram.

        Args:
            models (numpy.ndarray): Weighted posterior models.
            dep_int (numpy.ndarray, optional): Depth grid for interpolation.

        Returns:
            tuple: ``(fig, axes)`` containing the 2D histogram and auxiliary
            plots for interfaces.
        """
        if dep_int is None:
            dep_int = np.linspace(0, 100, 201)  # interppolate depth to 0.5 km.
            # bins for 2d histogram
            depbins = np.linspace(0, 100, 101)  # 1 km bins
        else:
            maxdepth = int(np.ceil(dep_int.max()))
            interp = dep_int[1] - dep_int[0]
            dep_int = np.arange(dep_int[0], dep_int[-1] + interp / 2., interp / 2.)
            depbins = np.arange(0, maxdepth + 2*interp, interp)  # interp km bins
            # nbin = np.arange(0, maxdepth + interp, interp)  # interp km bins

        # get interfaces, #first
        # end if
        models2 = ModelMatrix._replace_zvnoi_h(models)
        models2 = [model[~np.isnan(model)] for model in models2]
        yinterf = [np.cumsum(model[int(model.size/2):-1]) for model in models2]
        yinterf = np.concatenate(yinterf)

        vss_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)
        singlemodels = ModelMatrix.get_singlemodels(models, dep_int=depbins)

        vss_flatten = vss_int.flatten()
        vsinterval = 0.025  # km/s, 0.025 is assumption for vs_round
        # vsbins = int((vss_flatten.max() - vss_flatten.min()) / vsinterval)
        vs_histmin = vs_round(vss_flatten.min())-2*vsinterval
        vs_histmax = vs_round(vss_flatten.max())+3*vsinterval
        vsbins = np.arange(vs_histmin, vs_histmax, vsinterval) # some buffer

        # initiate plot
        fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},
                                 sharey=True, figsize=(10, 13))
        fig.subplots_adjust(wspace=0.05)

        data2d, xedges, yedges = np.histogram2d(vss_flatten, deps_int.flatten(),
                                				bins=(vsbins, depbins))

        axes[0].imshow(data2d.T, extent=(xedges[0], xedges[-1],
        							     yedges[0], yedges[-1]),
        			   origin='lower',
        			   vmax=len(models), aspect='auto')

        # plot mean / modes
        # colors = ['green', 'white']
        # for c, choice in enumerate(['mean', 'mode']):
        colors = ['white']
        for c, choice in enumerate(['mode']):
            vs, dep = singlemodels[choice]
            color = colors[c]
            axes[0].plot(vs, dep, color=color, lw=1, alpha=0.9, label=choice)

        # end for
        vs_mode, dep_mode = singlemodels['mode']
        axes[0].legend(loc=3)

        # histogram for interfaces
        data = axes[1].hist(yinterf, bins=depbins, orientation='horizontal',
                            color='lightgray', alpha=0.7,
                            edgecolor='k')
        bins, lay_bin = data[0].T, data[1].T
        center_lay = (lay_bin[:-1] + lay_bin[1:]) / 2.

        axes[0].set_ylabel('Depth in km')
        axes[0].set_xlabel('$V_S$ in km/s')

        axes[0].invert_yaxis()

        axes[0].set_title('%d models' % len(models))
        axes[1].set_xticks([])
        return fig, axes

    # end def _plot_bestmodels_hist
    def _get_posterior_data(self, data, final, chainidx=0):
        """Load posterior arrays for the requested dataset names.

        Args:
            data (list[str]): Dataset identifiers such as ``'models'`` or
                ``'likes'``.
            final (bool): If ``True``, load combined posterior files
                (``c_<name>.npy``). Otherwise pull the specified chain's phase
                2 arrays.
            chainidx (int, optional): Chain index to use when ``final`` is
                ``False``. Defaults to ``0``.

        Returns:
            list[numpy.ndarray]: Loaded arrays in the same order as ``data``.
        """
        if final:
            filetempl = op.join(self.datapath, 'c_%s.npy')
        else:
            filetempl = op.join(self.datapath, 'c%.3d_p2%s.npy' % (chainidx, '%s'))

        # end if
        outarrays = []
        for dataset in data:
            datafile = filetempl % dataset
            p2data = np.load(datafile)
            outarrays.append(p2data)

        # end for
        return outarrays

    # end def _get_posterior_data
    def get_models(self, data, final, chainidx=0):
        """Helper used by external tools to load posterior models.

        Args:
            data (list[str]): Dataset names; should include ``'models'``.
            final (bool): Whether to load combined posterior models.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            numpy.ndarray: Posterior model vectors.
        """
        models, = self._get_posterior_data(data, final, chainidx)
        return models

    # end def get_models
    def _plot_posterior_distribution(self, data, bins, formatter='%.2f', ax=None):
        """Draw and annotate a univariate posterior histogram.

        Args:
            data (numpy.ndarray): Samples to histogram.
            bins (int or array-like): Histogram bins.
            formatter (str, optional): Formatting string for annotations.
            ax (matplotlib.axes.Axes, optional): Axis to draw on; created when
                ``None``.

        Returns:
            matplotlib.axes.Axes: Axis with the histogram.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        # end if
        count, bins, _ = ax.hist(data, bins=bins, color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)
        cbins = (bins[:-1] + bins[1:]) / 2.
        mode = cbins[np.argmax(count)]
        median = np.median(data)

        if formatter is not None:
            text = 'median: %s' % formatter % median
            ax.text(0.97, 0.97, text,
                    fontsize=9, color='k',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes)

        # end if
        ax.axvline(median, color='k', ls=':', lw=1)
        
        # xticks = np.array(ax.get_xticks())
        # ax.set_xticklabels(xticks, fontsize=8)
        ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return ax

    # end def _plot_posterior_distribution
    @tryexcept
    def plot_posterior_likes(self, final=True, chainidx=0):
        """Plot the marginal posterior likelihood distribution.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Figure containing the histogram.
        """
        likes, = self._get_posterior_data(['likes'], final, chainidx)
        bins = 20
        formatter = '%d'

        ax = self._plot_posterior_distribution(likes, bins, formatter)
        ax.set_xlabel('Likelihood')
        return ax.figure

    # end def plot_posterior_likes
    @tryexcept
    def plot_posterior_misfits(self, final=True, chainidx=0):
        """Plot marginal posterior misfits for each target.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Figure containing target-specific
            misfit histograms.
        """
        misfits, = self._get_posterior_data(['misfits'], final, chainidx)

        datasets = [misfit for misfit in misfits.T]
        datasets = datasets[:-1]  # excluding joint misfit
        bins = 20
        formatter = '%.2f'

        fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6))
        for i, data in enumerate(datasets):
            axes[i] = self._plot_posterior_distribution(data, bins, formatter, ax=axes[i])
            axes[i].set_xlabel('RMS misfit (%s)' % self.refs[i])

        # end for
        return fig

    # end def plot_posterior_misfits
    @tryexcept
    def plot_posterior_nlayers(self, final=True, chainidx=0):
        """Plot the posterior distribution of accepted layer counts.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Figure with the layer-count histogram.
        """
        models, = self._get_posterior_data(['models'], final, chainidx)

        # get interfaces
        models = [model[~np.isnan(model)] for model in models]
        layers = np.array([(model.size/2 - 1) for model in models])

        bins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatter = '%d'
        ax = self._plot_posterior_distribution(layers, bins, formatter)

        xticks = np.arange(layers.min(), layers.max()+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('Number of layers')
        return ax.figure

    # end def plot_posterior_nlayers
    @tryexcept
    def plot_posterior_vpvs(self, final=True, chainidx=0):
        """Plot the marginal posterior of vp/vs.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Figure containing the vp/vs histogram.
        """
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)
        bins = 20
        formatter = '%.2f'

        ax = self._plot_posterior_distribution(vpvs, bins, formatter)
        ax.set_xlabel('$V_P$ / $V_S$')
        return ax.figure

    # end def plot_posterior_vpvs
    @tryexcept
    def plot_posterior_noise(self, final=True, chainidx=0):
        """Plot posterior distributions of noise hyper-parameters.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Figure with one histogram per noise
            parameter.
        """
        noise, = self._get_posterior_data(['noise'], final, chainidx)
        label = np.concatenate([['correlation (%s)' % ref, '$\sigma$ (%s)' % ref]
                               for ref in self.refs[:-1]])

        pars = int(len(noise.T)/2)
        fig, axes = plt.subplots(pars, 2, figsize=(14, 6*pars))
        fig.subplots_adjust(hspace=0.2)

        for i, data in enumerate(noise.T):
            if self.ntargets > 1:
                ax = axes[int(i/2)][i % 2]
            else:
                ax = axes[i % 2]

            # end if
            if np.std(data) == 0:  # constant during inversion
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                bins = 20
                formatter = '%.4f'
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
            # end if
            ax.set_xlabel(label[i])
        # end for
        return fig

    # end def plot_posterior_noise
    @tryexcept
    def plot_posterior_others(self, final=True, chainidx=0):
        """Plot a dashboard of likelihood, misfit, vp/vs, and layer marginals.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.

        Returns:
            matplotlib.figure.Figure: Multi-panel figure with histograms.
        """
        likes, = self._get_posterior_data(['likes'], final, chainidx)

        misfits, = self._get_posterior_data(['misfits'], final, chainidx)
        misfits = misfits.T[-1]  # only joint misfit
        vpvs, = self._get_posterior_data(['vpvs'], final, chainidx)

        models, = self._get_posterior_data(['models'], final, chainidx)
        models = np.array([model[~np.isnan(model)] for model in models])
        layers = np.array([(model.size/2 - 1) for model in models])
        nbins = np.arange(np.min(layers), np.max(layers)+2)-0.5

        formatters = ['%d', '%.2f', '%.2f', '%d']
        nbins = [20, 20, 20, nbins]
        labels = ['Likelihood', 'Joint misfit', '$V_P$ / $V_S$', 'Number of layers']

        fig, axes = plt.subplots(2, 2, figsize=(7, 6))
        axes = axes.flatten()
        for i, data in enumerate([likes, misfits, vpvs, layers]):
            ax = axes[i]

            if i == 2 and np.std(data) == 0:  # constant vpvs
                m = np.mean(data)
                bins = [m-1, m-0.1, m+0.1, m+1]
                formatter = None
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)
                ax.text(0.5, 0.5, 'constant: %.2f' % m, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_xticks([])
            else:
                formatter = formatters[i]
                bins = nbins[i]
                ax = self._plot_posterior_distribution(data, bins, formatter, ax=ax)

                if i == 3:
                    xticks = np.arange(layers.min(), layers.max()+1)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticks)

                # end if
            # end if
            ax.set_xlabel(labels[i])
        # end for
        return ax.figure

    # end def plot_posterior_others
    @tryexcept
    def plot_posterior_models1d(self, final=True, chainidx=0, depint=1):
        """Plot line-based summaries (mean/median/std) of the posterior models.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Specific chain when ``final`` is ``False``.
            depint (float, optional): Depth interpolation in km for resampling.

        Returns:
            matplotlib.figure.Figure: Figure containing the line plot.
        """
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        # end if
        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)
        fig, ax = self._plot_bestmodels(models, dep_int)
        # ax.set_xlim(self.priors['vs'])
        ax.set_ylim(self.priors['z'][::-1])
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        ax.set_title('%d models from %d chains' % (len(models), nchains))
        return fig

    #@tryexcept
    # end def plot_posterior_models1d
    def plot_posterior_models2d(self, final=True, chainidx=0, depint=1):
        """Plot 2D histograms of posterior vs-depth models.

        Args:
            final (bool, optional): Use combined posterior files if ``True``.
            chainidx (int, optional): Chain index when ``final`` is ``False``.
            depint (float, optional): Depth interpolation step in km.

        Returns:
            matplotlib.figure.Figure: Figure containing the histogram panels.
        """
        if final:
            nchains = self.initparams['nchains'] - self.outliers.size
        else:
            nchains = 1

        # end if
        models, = self._get_posterior_data(['models'], final, chainidx)

        dep_int = np.arange(self.priors['z'][0],
                            self.priors['z'][1] + depint, depint)

        fig, axes = self._plot_bestmodels_hist(models, dep_int)
        # axes[0].set_xlim(self.priors['vs'])
        axes[0].set_ylim(self.priors['z'][::-1])
        axes[0].set_title('%d models from %d chains' % (len(models), nchains))
        return fig


# Plot moho depth - crustal vs tradeoff

    # end def plot_posterior_models2d
    @tryexcept
    def plot_moho_crustvel_tradeoff(self, moho=None, mohovs=None, refmodel=None):
        """Analyze how Moho depth correlates with crustal velocity metrics.

        Args:
            moho (tuple, optional): Depth bounds (km) defining the Moho search
                window. Defaults to ``self.priors['z']``.
            mohovs (float, optional): Minimum velocity identifying the Moho.
                Defaults to ``4.2`` km/s.
            refmodel (tuple, optional): Reference interfaces (depth, velocity)
                to compare against.

        Returns:
            matplotlib.figure.Figure: Multi-panel histogram of Moho-related
            statistics.
        """
        models, vpvs = self._get_posterior_data(['models', 'vpvs'], final=True)

        if moho is None:
            moho = self.priors['z']
        # end if
        if mohovs is None:
            mohovs = 4.2  # km/s

        # end if
        mohos = np.zeros(len(models)) * np.nan
        vscrust = np.zeros(len(models)) * np.nan
        vslastlayer = np.zeros(len(models)) * np.nan
        vsjumps = np.zeros(len(models)) * np.nan

        for i, model in enumerate(models):
            thisvpvs = vpvs[i]
            vp, vs, h = Model.get_vp_vs_h(model, thisvpvs, self.mantle)
            # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
            # ifaces, vs = cdepth[1::2], cvs[::2]   # interfaces, vs
            ifaces = np.cumsum(h)
            vsstep = np.diff(vs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            if len(mohoidxs) == 0:
                continue

            # mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            # end if
            mohoidxs = mohoidxs.flatten()

            mohoidxs_vs = np.where((vs > mohovs))[0]-1
            if len(mohoidxs_vs) == 0:
                continue

            # end if
            mohoidx = np.intersect1d(mohoidxs, mohoidxs_vs)
            if len(mohoidx) == 0:
                continue
            # end if
            mohoidx = mohoidx[0]
            # ------

            thismoho = ifaces[mohoidx]
            crustmean = np.sum(vs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            lastvs = vs[mohoidx]
            vsjump = vsstep[mohoidx]

            mohos[i] = thismoho
            vscrust[i] = crustmean
            vslastlayer[i] = lastvs
            vsjumps[i] = vsjump

        # exclude nan values
        # end for
        mohos = mohos[~np.isnan(vsjumps)]
        vscrust = vscrust[~np.isnan(vsjumps)]
        vslastlayer = vslastlayer[~np.isnan(vsjumps)]
        vsjumps = vsjumps[~np.isnan(vsjumps)]

        fig, ax = plt.subplots(2, 4, figsize=(11, 6))
        fig.subplots_adjust(hspace=0.05)
        fig.subplots_adjust(wspace=0.05)

        labels = ['$V_S$ last crustal layer', '$V_S$ crustal mean', '$V_S$ increase']
        bins = 50

        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                histdata = ax[0][n].hist(xdata, bins=bins,
                                         color='darkblue', alpha=0.7,
                                         edgecolor='white', linewidth=0.4)

                median = np.median(xdata)
                ax[0][n].axvline(median, color='k', ls='--', lw=1.2, alpha=1)
                stats = 'median:\n%.2f km/s' % median
                ax[0][n].text(0.97, 0.97, stats,
                              fontsize=9, color='k',
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform=ax[0][n].transAxes)
            except:
                pass

        # end for
        for n, xdata in enumerate([vslastlayer, vscrust, vsjumps]):
            try:
                ax[1][n].set_xlabel(labels[n])

                data = ax[1][n].hist2d(xdata, mohos, bins=bins)
                data2d, xedges, yedges, _ = np.array(data).T

                xi, yi = np.unravel_index(data2d.argmax(), data2d.shape)
                x_mode = ((xedges[:-1] + xedges[1:]) / 2.)[xi]
                y_mode = ((yedges[:-1] + yedges[1:]) / 2.)[yi]

                ax[1][n].axhline(y_mode, color='white', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(x_mode, color='white', ls='--', lw=0.5, alpha=0.7)

                xmin, xmax = ax[1][n].get_xlim()
                ax[0][n].set_xlim([xmin, xmax])
            except:
                pass

            ax[0][n].set_yticks([])
            ax[0][n].set_yticklabels([], visible=False)
            ax[0][n].set_xticklabels([], visible=False)

        # end for
        ax[1][1].set_yticklabels([], visible=False)
        ax[1][2].set_yticklabels([], visible=False)
        ax[1][3].set_yticklabels([], visible=False)
        ax[1][0].set_ylabel('Moho depth in km')

        # plot moho 1d histogram
        histdata = ax[1][3].hist(mohos, bins=bins, orientation='horizontal',
                                 color='darkblue', alpha=0.7,
                                 edgecolor='white', linewidth=0.4)

        median = np.median(mohos)
        std = np.std(mohos)
        print('moho: %.4f +- %.4f km' % (median, std))
        ax[1][3].axhline(median, color='k', ls='--', lw=1.2, alpha=1)
        stats = 'median:\n%.2f km' % median
        ax[1][3].text(0.97, 0.97, stats,
                      fontsize=9, color='k',
                      horizontalalignment='right',
                      verticalalignment='top',
                      transform=ax[1][3].transAxes)
        ymin, ymax = ax[1][0].get_ylim()
        # ymin, ymax = median - 4*std, median + 4*std
        ax[1][0].set_ylim(ymin, ymax)
        ax[1][1].set_ylim(ymin, ymax)
        ax[1][2].set_ylim(ymin, ymax)
        ax[1][3].set_ylim(ymin, ymax)

        ax[1][3].set_xticklabels([], visible=False)
        ax[1][3].set_yticks([])
        ax[1][3].set_yticklabels([], visible=False)
        ax[0][3].axis('off')

        if refmodel is not None:
            dep, vs = refmodel
            h = (dep[1:] - dep[:-1])[::2]
            ifaces, lvs = dep[1::2], vs[::2]

            vsstep = np.diff(lvs)  # velocity change at interfaces
            mohoidxs = np.argwhere((ifaces > moho[0]) & (ifaces < moho[1]))
            mohoidx = mohoidxs[np.argmax(vsstep[mohoidxs])][0]
            truemoho = ifaces[mohoidx]
            truecrust = np.sum(lvs[:(mohoidx+1)] * h[:(mohoidx+1)]) / ifaces[mohoidx]
            truevslast = lvs[mohoidx]
            truevsjump = vsstep[mohoidx]

            for n, xdata in enumerate([truevslast, truecrust, truevsjump]):
                ax[1][n].axhline(truemoho, color='red', ls='--', lw=0.5, alpha=0.7)
                ax[1][n].axvline(xdata, color='red', ls='--', lw=0.5, alpha=0.7)

            # end for
        # end if
        return fig

# Plot current models and data fits. also plot best data fit incl. model.

    # end def plot_moho_crustvel_tradeoff
    @tryexcept
    def plot_currentmodels(self, nchains):
        """Plot step models for the latest accepted state of each chain.

        Args:
            nchains (int): Number of chains to visualize regardless of outlier
                status.

        Returns:
            matplotlib.figure.Figure: Figure containing velocity-depth curves.
        """
        fig, ax = plt.subplots(figsize=(4, 6.5))

        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            color = color_list[i]
            vp, vs, h = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)

            label = 'c%d / %d' % (chainidx, vs.size-1)
            ax.plot(cvs, cdepth, color=color, ls='-', lw=0.8,
                    alpha=0.7, label=label)

        # end for
        ax.invert_yaxis()
        ax.set_xlabel('$V_S$ in km/s')
        ax.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax.set_ylim(self.priors['z'][::-1])
        ax.set_title('Current models')
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return fig

    # end def plot_currentmodels
    @tryexcept
    def plot_currentdatafits(self, nchains):
        """Plot current synthetic data fits for the first ``nchains`` chains.

        Args:
            nchains (int): Number of chains to visualize regardless of outlier
                status.

        Returns:
            matplotlib.figure.Figure: Figure comparing observed and modeled
            data per target.
        """
        base = cm.get_cmap(name='rainbow')
        color_list = base(np.linspace(0, 1, nchains))
        targets = Targets.JointTarget(targets=self.targets)

        fig, ax = targets.plot_obsdata(mod=False)

        for i, modfile in enumerate(self.modfiles[1][:nchains]):
            color = color_list[i]
            chainidx, _, _ = self._return_c_p_t(modfile)
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            currentvpvs = vpvs[-1]
            currentmodel = models[-1]

            vp, vs, h = Model.get_vp_vs_h(currentmodel, currentvpvs, self.mantle)
            rho = vp * 0.32 + 0.77

            jmisfit = 0
            for n, target in enumerate(targets.targets):
                xmod, ymod = target.moddata.plugin.run_model(
                    h=h, vp=vp, vs=vs, rho=rho)

                yobs = target.obsdata.y
                misfit = target.valuation.get_rms(yobs, ymod)
                jmisfit += misfit

                label = ''
                if len(targets.targets) > 1:
                    if ((len(targets.targets) - 1) - n) < 1e-2:
                        label = 'c%d / %.3f' % (chainidx, jmisfit)
                    # end if
                    ax[n].plot(xmod, ymod, color=color, alpha=0.7, lw=0.8,
                               label=label)
                else:
                    label = 'c%d / %.3f' % (chainidx, jmisfit)
                    ax.plot(xmod, ymod, color=color, alpha=0.5, lw=0.7,
                            label=label)

                # end if
            # end for
        # end for
        if len(targets.targets) > 1:
            ax[0].set_title('Current data fits')
            idx = len(targets.targets) - 1
            han, lab = ax[idx].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Current data fits')
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        # end if
        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    # end def plot_currentdatafits
    @tryexcept
    def plot_bestmodels(self):
        """Plot the best-fitting model encountered per non-outlier chain.

        Returns:
            matplotlib.figure.Figure: Figure showing velocity-depth curves for
            each best-fitting model.
        """
        fig, ax = plt.subplots(figsize=(4, 6.5))

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]

        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            # end if
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            # end if
            vp, vs, h = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)

            ax.plot(cvs, cdepth, color='k', ls='-', lw=0.8, alpha=0.5)

        # label = 'c%d' % thebestchain
        # vp, vs, h = Model.get_vp_vs_h(thebestmodel, thebestvpvs, self.mantle)
        # cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
        # ax.plot(cvs, cdepth, color='red', ls='-', lw=1,
        #         alpha=0.8, label=label)

        # end for
        ax.invert_yaxis()
        ax.set_xlabel('$V_S$ in km/s')
        ax.set_ylabel('Depth in km')
        # ax.set_xlim(self.priors['vs'])
        ax.set_ylim(self.priors['z'][::-1])
        ax.set_title('Best fit models from %d chains' %
                     (len(modfiles)-self.outliers.size))
        ax.grid(color='gray', alpha=0.6, ls=':', lw=0.5)
        # ax.legend(loc=3)
        return fig

    # end def plot_bestmodels
    @tryexcept
    def plot_bestdatafits(self):
        """Plot synthetic data from the best-fitting model of each chain.

        Returns:
            matplotlib.figure.Figure: Figure containing data fits per target.
        """
        targets = Targets.JointTarget(targets=self.targets)
        fig, ax = targets.plot_obsdata(mod=False)

        thebestmodel = np.nan
        thebestmisfit = 1e15
        thebestchain = np.nan

        modfiles = self.modfiles[1]

        for i, modfile in enumerate(modfiles):
            chainidx, _, _ = self._return_c_p_t(modfile)
            if chainidx in self.outliers:
                continue
            # end if
            models = np.load(modfile)
            vpvs = np.load(modfile.replace('models', 'vpvs')).T
            misfits = np.load(modfile.replace('models', 'misfits')).T[-1]
            bestmodel = models[np.argmin(misfits)]
            bestvpvs = vpvs[np.argmin(misfits)]
            bestmisfit = misfits[np.argmin(misfits)]

            if bestmisfit < thebestmisfit:
                thebestmisfit = bestmisfit
                thebestmodel = bestmodel
                thebestvpvs = bestvpvs
                thebestchain = chainidx

            # end if
            vp, vs, h = Model.get_vp_vs_h(bestmodel, bestvpvs, self.mantle)
            rho = vp * 0.32 + 0.77

            for n, target in enumerate(targets.targets):
                xmod, ymod = target.moddata.plugin.run_model(
                    h=h, vp=vp, vs=vs, rho=rho)

                if len(targets.targets) > 1:
                    ax[n].plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)
                else:
                    ax.plot(xmod, ymod, color='k', alpha=0.5, lw=0.7)

                # end if
            # end for
        # end for
        if len(targets.targets) > 1:
            ax[0].set_title('Best data fits from %d chains' %
                            (len(modfiles)-self.outliers.size))
            # idx = len(targets.targets) - 1
            han, lab = ax[0].get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax[0].legend().set_visible(False)
        else:
            ax.set_title('Best data fits from %d chains' %
                         (len(modfiles)-self.outliers.size))
            han, lab = ax.get_legend_handles_labels()
            handles, labels = self._unique_legend(han, lab)
            ax.legend().set_visible(False)

        # end if
        fig.legend(handles, labels, loc='center left',
                   bbox_to_anchor=(0.92, 0.5))
        return fig

    # end def plot_bestdatafits
    @tryexcept
    def plot_rfcorr(self, rf='prf'):
        """Compare receiver-function residuals with noise realizations.

        Args:
            rf (str, optional): Receiver-function name present in ``self.refs``.

        Returns:
            matplotlib.figure.Figure: Figure showing residuals and sampled
            noise.
        """
        from BayHunter import SynthObs

        p2models, p2noise, p2misfits, p2vpvs = self._get_posterior_data(
            ['models', 'noise', 'misfits', 'vpvs'], final=True)

        fig, axes = plt.subplots(2, sharex=True, sharey=True)
        ind = self.refs.index(rf)
        best = np.argmin(p2misfits.T[ind])
        model = p2models[best]
        vpvs = p2vpvs[best]

        target = self.targets[ind]
        x, y = target.obsdata.x, target.obsdata.y
        vp, vs, h = Model.get_vp_vs_h(model, vpvs, self.mantle)
        rho = vp * 0.32 + 0.77

        _, ymod = target.moddata.plugin.run_model(
            h=h, vp=vp, vs=vs, rho=rho)
        yobs = target.obsdata.y
        yresiduals = yobs - ymod

        # axes[0].set_title('Residuals [dobs-g(m)] obtained with best fitting model m')
        axes[0].plot(x, yresiduals, color='k', lw=0.7, label='residuals')

        corr, sigma = p2noise[best][2*ind:2*(ind+1)]
        yerr = SynthObs.compute_gaussnoise(y, corr=corr, sigma=sigma)
        # axes[1].set_title('One Realization of random noise from inferred CeRF')
        axes[1].plot(x, yerr, color='k', lw=0.7, label='noise realization')
        axes[1].set_xlabel('Time in s')

        axes[0].legend(loc=4)
        axes[1].legend(loc=4)
        axes[0].grid(color='gray', ls=':', lw=0.5)
        axes[1].grid(color='gray', ls=':', lw=0.5)
        axes[0].set_xlim([x[0], x[-1]])

        return fig

    # end def plot_rfcorr
    def merge_pdfs(self):
        """Combine all plot PDFs into a single ``c_summary.pdf`` artifact.

        Returns:
            None
        """
        from PyPDF2 import PdfReader, PdfWriter

        outputfile = op.join(self.figpath, 'c_summary.pdf')
        output = PdfWriter()
        pdffiles = glob.glob(op.join(self.figpath + os.sep + 'c_*.pdf'))
        pdffiles.sort(key=op.getmtime)

        for pdffile in pdffiles:
            if pdffile == outputfile:
                continue

            # end if
            document = PdfReader(open(pdffile, 'rb'))
            for i in range(len(document.pages)):
                output.add_page(document.pages[i])

            # end for
        # end for
        with open(outputfile, "wb") as f:
            output.write(f)

        # end with
    # end def merge_pdfs
    def save_chainplots(self, cidx=0, refmodel=dict(), depint=None):
        """Save posterior summary plots for a specific chain.

        Args:
            cidx (int, optional): Chain index to visualize.
            refmodel (dict, optional): Dictionary with optional reference
                entries (``model``, ``nlays``, ``noise``) to overlay.
            depint (float, optional): Depth interpolation step in km.

        Returns:
            None
        """
        self.refmodel.update(refmodel)
        # plot chain specific posterior distributions

        fig5a = self.plot_posterior_misfits(final=False, chainidx=cidx)
        self.savefig(fig5a, 'c%.3d_posterior_misfit.pdf' % cidx)

        fig5b = self.plot_posterior_nlayers(final=False, chainidx=cidx)
        self.plot_refmodel(fig5b, 'nlays')
        self.savefig(fig5b, 'c%.3d_posterior_nlayers.pdf' % cidx)

        fig5c = self.plot_posterior_noise(final=False, chainidx=cidx)
        self.plot_refmodel(fig5c, 'noise')
        self.savefig(fig5c, 'c%.3d_posterior_noise.pdf' % cidx)

        fig5d = self.plot_posterior_models1d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5d, 'model', color='k', lw=1)
        self.savefig(fig5d, 'c%.3d_posterior_models1d.pdf' % cidx)

        fig5e = self.plot_posterior_models2d(
            final=False, chainidx=cidx, depint=depint)
        self.plot_refmodel(fig5e, 'model', color='red', lw=0.5, alpha=0.7)
        self.savefig(fig5e, 'c%.3d_posterior_models2d.pdf' % cidx)

    # end def save_chainplots
    def save_plots(self, nchains=5, refmodel=dict(), depint=1):
        """Save the default set of diagnostic plots for all chains.

        Args:
            nchains (int, optional): Number of chains shown in iteration plots.
            refmodel (dict, optional): Optional reference dictionary used by
                ``plot_refmodel``.
            depint (float, optional): Depth interpolation step in km.

        Returns:
            None
        """
        self.refmodel.update(refmodel)

        nchains = np.min([nchains, len(self.likefiles[1])])

        # plot values changing over iteration
        # fig1a = self.plot_iiterlikes(nchains=nchains)
        # self.savefig(fig1a, 'c_iiter_likes.png')

        fig1b = self.plot_iitermisfits(nchains=nchains, ind=-1)
        self.savefig(fig1b, 'c_iiter_misfits.png')

        # fig1c = self.plot_iiternlayers(nchains=nchains)
        # self.savefig(fig1c, 'c_iiter_nlayers.png')
        #
        # fig1d = self.plot_iitervpvs(nchains=nchains)
        # self.savefig(fig1d, 'c_iiter_vpvs.png')
        #
        # for i in range(self.ntargets):
        #     ind = i * 2 + 1
        #     fig1d = self.plot_iiternoise(nchains=nchains, ind=ind)
        #     self.savefig(fig1d, 'c_iiter_noisepar%d.png' % ind)

        # plot current models and datafit
        # end for
        fig3a = self.plot_currentmodels(nchains=nchains)
        self.plot_refmodel(fig3a, 'model', color='k', lw=1)
        self.savefig(fig3a, 'c_currentmodels.png')

        # fig3b = self.plot_currentdatafits(nchains=nchains)
        # self.savefig(fig3b, 'c_currentdatafits.png')
        #
        # # plot final posterior distributions
        # fig2b = self.plot_posterior_nlayers()
        # self.plot_refmodel(fig2b, 'nlays')
        # self.savefig(fig2b, 'c_posterior_nlayers.png')
        #
        # fig2b = self.plot_posterior_vpvs()
        # self.plot_refmodel(fig2b, 'vpvs')
        # self.savefig(fig2b, 'c_posterior_vpvs.png')
        #
        # fig2c = self.plot_posterior_noise()
        # self.plot_refmodel(fig2c, 'noise')
        # self.savefig(fig2c, 'c_posterior_noise.png')
        #
        # fig2d = self.plot_posterior_models1d(depint=depint)
        # self.plot_refmodel(fig2d, 'model', color='k', lw=1)
        # self.savefig(fig2d, 'c_posterior_models1d.png')
        # fig2e = self.plot_posterior_models2d(depint=depint)
        # self.plot_refmodel(fig2e, 'model', color='red', lw=0.5, alpha=0.7)
        # self.savefig(fig2e, 'c_posterior_models2d.png')
    # end def save_plots
# end class PlotFromStorage
