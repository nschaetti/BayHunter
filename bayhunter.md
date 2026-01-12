# BayHunter technical guide

## Inversion workflow and algorithmic flow

### 1. Parameterisation and priors
- **Voronoi parameterisation (`BayHunter.Models.Model`)**  
  Layered velocity structures are represented by Voronoi nuclei: an n-layer model is stored as `[vs_1..vs_n, z_1..z_n]`. Helper methods such as `split_modelparams`, `get_vp_vs_h`, and `get_stepmodel` turn those vectors into physical layer thicknesses, P-wave velocities (via the vp/vs ratio), densities, and step-wise profiles for plotting or histogramming.
- **Prior handling (`BayHunter.SingleChain.draw_initmodel`, `draw_initvpvs`, `draw_initnoiseparams`)**  
  Initial models are drawn from ranges prescribed in `defaults.ini` (or user overrides). Depth nodes are optionally constrained around Moho estimates, vs values are sampled uniformly and sorted to enforce monotonic trends, vp/vs ratios can be fixed or sampled, and correlated noise hyper-parameters per target are initialised from their priors.

### 2. Targets, forward models, and likelihood
- **Targets (`BayHunter.Targets.SingleTarget`, subclasses, and `JointTarget`)**  
  A `JointTarget` bundles multiple `SingleTarget` objects. Each single target couples observed data (`ObservedData`) with a forward model (`ModeledData`) implemented by wrapper classes (`RFminiModRF` for receiver functions and `SurfDisp` for dispersion curves). During evaluation, `JointTarget.evaluate` converts the current model into layer thicknesses, runs the forward engines to get synthetic data, computes Mahalanobis misfits using covariance matrices provided by `Valuation`, and sums log-likelihood contributions across targets.
- **Noise covariance physics (`Valuation`)**  
  The likelihood is fully Bayesian rather than ABC-style: the code constructs analytical covariance matrices based on diagonal, exponential, or Gaussian correlation laws (`get_covariance_*`) depending on the target noise reference and uses them to compute `logL = -½[(y-ŷ)^T C⁻¹ (y-ŷ) + log|C| + n log 2π]`.

### 3. Proposal moves and acceptance (`BayHunter.SingleChain`)
- **Move set**  
  `SingleChain.iterate` randomly picks a move type:
  - `vsmod` / `zvmod`: Gaussian perturbations on a single velocity or Voronoi depth (`_model_vschange`, `_model_zvnoi_move`).
  - `birth` / `death`: Reversible-jump moves that insert or remove nuclei (`_model_layerbirth`, `_model_layerdeath`), changing the dimensionality of the model.
  - `noise`: Gaussian proposal on a free noise hyper-parameter (`_get_hyperparameter_proposal`).
  - `vpvs`: Gaussian proposal on the vp/vs ratio (`_get_vpvs_proposal`).
- **Physical validity checks**  
  `_validmodel`, `_validnoise`, and `_validvpvs` enforce geophysical constraints: layer count inside priors, minimum thickness (`thickmin`), vs bounds, interface depth bounds, low/high velocity zone percentages, and noise ranges. Invalid proposals are rejected before forward runs.
- **Likelihood evaluation → acceptance**  
  Valid proposals trigger `Model.get_vp_vs_h`, `JointTarget.evaluate`, and log-likelihood storage. Metropolis-Hastings acceptance is done in log space with `u = log(rand())` and `alpha = ΔlogL` for fixed-dimensional moves. Birth/death moves use the Bodin et al. (2012) RJMCMC formulation implemented in `get_acceptance_probability`, where Jacobian and proposal-density ratios appear as `log(A)` and `±B` terms built from the new vs difference and proposal variance.
- **Adaptive proposal widths**  
  `adjust_propdist` monitors proposal acceptance ratios and rescales the proposal standard deviations every 1000 iterations to maintain the user-specified acceptance window.
- **Weighting and thinning**  
  Accepted states are appended via `append_currentmodel`; unaccepted iterations implicitly weight the last accepted model by counting how many iterations it stays current. After burn-in (`iter_burnin`) and main (`iter_main`) phases, `get_weightedvalues` replicates models according to this weight so that posterior statistics retain time-in-chain information. `save_finalmodels` writes numpy archives split into burn-in and posterior phases, thinned to respect `maxmodels`.

### 4. Multi-chain orchestration (`BayHunter.mcmcOptimizer.MCMC_Optimizer`)
- Shared-memory arrays (`_init_shareddata`) store models, misfits, likelihoods, noise parameters, and vp/vs ratios for all chains.  
- `_init_chain` instantiates `SingleChain` objects that share those arrays, enabling post-processing without reloading from disk.  
- `mp_inversion` launches multiple processes (up to CPU count or `nthreads`) and can spawn `monitor_process` to stream the latest accepted model/likelihood/noise tuples to BayWatch live plots.  
- After all chains finish, thin/weighted results per chain are saved. Optional managers allow retrieving chain objects for interactive analysis.

### 5. Monitoring and plotting
- **BayWatch (`BayHunter.BayWatch.BayWatcher`)** listens to the ZeroMQ stream and live-plots the most recent models, vp/vs, likelihoods, noise parameters, and forward fits.  
- **Plotting (`BayHunter.Plotting.PlotFromStorage`)** ingests the `.npy` archives, detects outlier chains, merges posterior ensembles, and offers a wide range of plotting utilities (iteration-wise diagnostics, posterior histograms, Vs-depth fan plots, data fit overlays, etc.).  
- **Synthetic experiments (`BayHunter.SynthObs.SynthObs`)** generate reference dispersion/RF data and correlated noise so you can validate the inversion setup or compute expected likelihoods for BayWatch reference lines.

## How BayHunter differs from classical ABC
Classical Approximate Bayesian Computation (ABC) bypasses explicit likelihoods. It perturbs priors, simulates data, and accepts/rejects samples based on whether a distance between observed and simulated summary statistics falls below an ε threshold. This introduces two approximations: (1) summary statistics may not be sufficient, and (2) the tolerance ε biases the posterior unless ε→0 with huge simulation cost.

BayHunter instead computes the **exact Gaussian log-likelihood** for each target. The code evidences this in `JointTarget.evaluate`, where synthetic data and observed data enter the Mahalanobis distance with physically-derived covariance matrices (`Valuation.get_covariance_*`). Acceptance uses the true likelihood ratio (plus Jacobian terms for RJ moves). There is no ε tolerance, and forward simulations are only run for valid proposals. Additionally, BayHunter samples noise hyper-parameters jointly with structural parameters, while ABC typically treats noise heuristically. Consequently, BayHunter delivers posterior samples that are asymptotically exact for the stated forward model and noise assumptions, whereas ABC delivers approximate samples whose accuracy depends on distance metric design.

## Module map

| Module | Role |
| --- | --- |
| `BayHunter.Models` | Converts between Voronoi parameter vectors and geophysical quantities (vp, vs, thickness, interpolated profiles). |
| `BayHunter.Targets` | Encapsulates observations, forward-model plugins (`RFmini`, `SurfDisp`), and likelihood evaluation for single and joint targets. |
| `BayHunter.SingleChain` | Implements one RJMCMC chain: model/noise/vpvs proposals, acceptance logic, adaptive proposals, storage. |
| `BayHunter.mcmcOptimizer` | Multi-chain orchestration, shared-memory storage, BayWatch streaming. |
| `BayHunter.Plotting` | Offline diagnostics and posterior visualisation from saved arrays. |
| `BayHunter.BayWatch` | Real-time plotter fed by the monitor socket. |
| `BayHunter.SynthObs` | Synthetic data generators and analytic noise/likelihood utilities. |
| `BayHunter.utils` | Config helpers, serialization utilities, RF noise diagnostics. |
| `BayHunter.rfmini_modrf` / `surf96_modsw` | Python wrappers around compiled RFmini and Surf96 forward solvers. |
| `BayHunter.data.*` | Dataset-generation API (priors, samples, dispersion curves, Arrow/parquet export, HuggingFace helpers). |
| `BayHunter.__main__` | Click-based CLI for plotting perturbations, running forward models, sampling priors, and generating datasets. |

## Class and method reference (scientific / mathematical behaviour)

### Core inversion primitives

#### `BayHunter.Models.Model`
- `split_modelparams(model)`: splits a packed vector into vs and Voronoi depths (nuclei) by removing NaNs.
- `get_vp(vs, vpvs=1.73, mantle=[4.3, 1.8])`: applies crustal vp/vs and mantle overrides to compute P-wave velocities.
- `get_vp_vs_h(model, vpvs=1.73, mantle=None)`: converts Voronoi nodes into vp, vs, and layer thickness vector `h` by locating discontinuities midway between nuclei.
- `get_stepmodel(model, vpvs=1.73, mantle=None)`: returns stepwise vp/vs vs depth arrays for plotting; sets half-space depth.
- `get_stepmodel_from_h(h, vs, vpvs=1.73, dep=None, vp=None, mantle=None)`: same as above but when `h` is known directly.
- `get_interpmodel(model, dep_int, vpvs=1.73, mantle=None)`: interpolates the step model onto a regular depth grid for histograms.

#### `BayHunter.Models.ModelMatrix`
- `_delete_nanmodels(models)`: removes rows containing NaNs before statistics.  
- `_replace_zvnoi_h(models)`: maps each model’s z-voronoi parametrization to (vs, h) pairs for plotting.  
- `get_interpmodels(models, dep_int)`: returns interpolated vs-depth grids for all models.  
- `get_singlemodels(models, dep_int=None, misfits=None)`: derives summary models (mean, median, minmax, std envelopes, histogram mode, optional best misfit).  
- `get_weightedvalues(weights, models=None, likes=None, misfits=None, noiseparams=None, vpvs=None)`: repeats rows according to integer weights, effectively applying time-in-chain replication to arrays.

#### `BayHunter.Targets`
- `ObservedData.__init__(x, y, yerr=None)`: stores observation vectors and enforces sane error defaults.  
- `ModeledData.__init__(obsx, ref)`: selects the appropriate forward plugin (RFmini or SurfDisp) based on target reference and preallocates arrays.  
- `ModeledData.update(plugin)`: swaps to a user-supplied forward model.  
- `ModeledData.calc_synth(h, vp, vs, **kwargs)`: run plugin with densities and optional hyperparameters to produce synthetic x,y.  
- `Valuation.__init__`: caches covariance inverses/determinants.  
- `Valuation.get_rms`: RMS misfit for logging.  
- `Valuation.get_covariance_nocorr`, `get_covariance_nocorr_scalederr`, `get_covariance_exp`, `init_covariance_gauss`, `get_covariance_gauss`: compute covariance matrices/inverses/log-determinants under different correlation assumptions.  
- `Valuation.get_likelihood(yobs, ymod, c_inv, logc_det)`: Gaussian log-likelihood, i.e. Mahalanobis distance with normalization.  
- `SingleTarget.__init__(x, y, ref, yerr=None)`: binds observation, forward engine, and valuation object; the `ref` string ties into noise priors.  
- `SingleTarget.update_plugin`, `_moddata_valid`, `calc_misfit`, `calc_likelihood`, `plot`: housekeeping for forward results and diagnostic plots.  
- `RayleighDispersionPhase`/`RayleighDispersionGroup`/`LoveDispersionPhase`/`LoveDispersionGroup`/`PReceiverFunction`/`SReceiverFunction`: convenience wrappers that set `ref` and `noiseref`.  
- `JointTarget.__init__(targets)`: aggregates single targets.  
- `JointTarget.get_misfits()`: concatenates misfits plus joint sum.  
- `JointTarget.evaluate(h, vp, vs, noise, **kwargs)`: runs forward models, validates, computes log-likelihood sum, stores proposals.  
- `JointTarget.plot_obsdata(ax=None, mod=False)`: quick-look plot of observed data (and optionally current model curves).

#### `BayHunter.SingleChain.SingleChain`
- `__init__(targets, ...)`: loads priors (`utils.load_params`), configures run lengths/acceptance windows, seeds RNG, attaches shared arrays, draws initial model/noise/vpvs, evaluates targets, and writes the first state.  
- `_init_model_and_currentvalues()`: draws and validates the starting triple (model, vpvs, noise), sets target-specific covariance routines, evaluates and stores the first log-likelihood.  
- `draw_initmodel()`: samples Vs and Voronoi depths respecting priors (including Moho constraints).  
- `draw_initnoiseparams()`: samples/flags per-target correlated-noise hyperparameters and records which ones stay fixed.  
- `draw_initvpvs()`: samples or returns fixed vp/vs prior.  
- `set_target_covariance(corrfix, noise_corr, rcond=None)`: chooses which `Valuation` covariance function each target should use (diagonal, exponential, or precomputed Gaussian).  
- `_init_chainarrays(...)`: maps the shared raw arrays onto numpy views for this chain (models, misfits, likes, noise, vpvs, iteration indices).  
- `_model_layerbirth` / `_model_layerdeath`: RJMCMC moves acting on Voronoi nuclei; they also compute squared velocity differences for Jacobian terms.  
- `_model_vschange` / `_model_zvnoi_move`: random-walk proposals for velocities or depths.  
- `_get_modelproposal(modify)`: dispatches to the appropriate proposal generator and sorts the result.  
- `_sort_modelproposal(model)`: ensures increasing depth order after perturbations.  
- `_validmodel(model)`: enforces layer count, thickness, Vs bounds, interface depth bounds, LVZ/HVZ percentage constraints using `Model.get_vp_vs_h`.  
- `_get_hyperparameter_proposal()` / `_validnoise(noise)`: propose and check Gaussian perturbations on unfixed noise parameters.  
- `_get_vpvs_proposal()` / `_validvpvs(vpvs)`: propose and bound-check vp/vs.  
- `adjust_propdist()`: rescales each proposal standard deviation by ±5% to keep acceptance within `[acceptance[0], acceptance[1]]`.  
- `get_acceptance_probability(modify)`: returns log α for the current move type, adding Bodin RJMCMC terms for birth/death.  
- `accept_as_currentmodel(model, noise, vpvs)`: commit the proposal as the new current state; update misfit/likelihood caches.  
- `append_currentmodel()`: write the current state into shared arrays, track iteration index, and bump acceptance counters.  
- `iterate()`: main logic per iteration (select move, get proposal, evaluate, accept/reject, log stats, adapt proposals).  
- `run_chain()`: drives `iterate` through burn-in+main phases, slices arrays to the number of accepted models, splits into phase 1/2 weighted distributions, computes thinning factor, and triggers saving.  
- `get_weightedvalues(pind, finaliter)`: calculates how many times each accepted state should be repeated based on iteration numbers to emulate the Markov chain occupancy.  
- `save_finalmodels()`: writes `.npy` files (`c###_p{1,2}{models,likes,misfits,noise,vpvs}`) to the station’s save path, applying thinning.

#### `BayHunter.mcmcOptimizer.MCMC_Optimizer`
- `__init__(targets, initparams, priors, random_seed)`: loads defaults, stores target & chain counts, precomputes shared arrays, instantiates each `SingleChain`, and readies a multiprocessing manager.  
- `_init_shareddata()`: allocates raw shared arrays for models, misfits, likelihoods, noise parameters, and vp/vs values, sized according to `iter_burnin+iter_main` and acceptance expectations.  
- `_init_chain(chainidx, targets)`: creates a `SingleChain` with pointers to shared arrays and a unique RNG seed.  
- `monitor_process(dtsend)`: publishes the latest accepted models/likelihoods/noise/vpvs over ZeroMQ to BayWatch, using helper functions that pick the newest non-NaN entries per chain.  
- `mp_inversion(baywatch=False, dtsend=0.5, nthreads=0)`: manages multiprocessing: optionally spawns `monitor_process`, dispatches chain indices to worker processes (respecting `nthreads`), waits for completion, collects results via a manager list, and logs runtime.

### Monitoring and plotting tools

#### `BayHunter.BayWatch.BayWatcher`
- `__init__`: connects to the PUB socket, loads config/prior/initparams for plotting ranges, allocates history buffers, and builds the interactive figure.  
- `init_style_dicts()`: defines matplotlib styles for RF/SWD/noise series.  
- `init_plot()`: builds the composite figure (Vs-depth, Rayleigh & RF fits, likelihood, noise, vp/vs inset) and initialises artists/collections.  
- `_same_event(event, eventtime)`: helper to de-bounce button clicks.  
- `next(event)` / `prev(event)`: cycle through stored models when manually browsing.  
- `update_chain()`: ingest the latest arrival, recompute synthetic data for the reference model, and update subplot data.  
- `compute_synth(h, vs, vp)`: recompute forward data fits for the panel overlays.  
- `init_arrays()`: reset internal buffers whenever a new chain starts streaming.  
- `store_data(arrmodels, arrlikes, arrnoise, arrvpvs)`: append new raw values to circular buffers.  
- `update_models`, `update_likes`, `update_vpvss`, `update_noises`: refresh the plotted elements to reflect the latest sample.  
- `watch()`: main loop reading from socket, throttling updates, and handling user events.  
- `main()`: CLI entry point that loads config and launches the watcher.

#### `BayHunter.Plotting.PlotFromStorage`
Key methods (all operate on saved `.npy` arrays):
- `__init__(configfile)`: load targets, priors, initparams, figure directory, file lists, and optional reference model.  
- `read_config`, `savefig`, `init_outlierlist`, `init_filelists`, `get_outliers`, `_get_chaininfo`: data management utilities.  
- `save_final_distribution(maxmodels, dev)`: detect outlier chains, thin per-chain samples, stack the posterior, and save combined arrays.  
- `_unique_legend`, `_return_c_p_t`, `_sort`, `_get_layers`: helper utilities for labeling and depth calculations.  
- `plot_refmodel`: overlay a provided reference model (velocity profile, noise parameters, vp/vs, or misfits).  
- `_plot_iitervalues`, `plot_iitermisfits`, `plot_iiterlikes`, `plot_iiternoise`, `plot_iiternlayers`, `plot_iitervpvs`: iteration-wise diagnostics of chain convergence.  
- `_plot_bestmodels`, `_plot_bestmodels_hist`, `_plot_posterior_distribution`: helper functions for summarising best models and histograms.  
- `_get_posterior_data`, `get_models`: load and optionally down-sample posterior arrays for specific chains.  
- `plot_posterior_likes`, `plot_posterior_misfits`, `plot_posterior_nlayers`, `plot_posterior_vpvs`, `plot_posterior_noise`, `plot_posterior_others`: histograms/CDFs of scalar quantities.  
- `plot_posterior_models1d`, `plot_posterior_models2d`: depth-Vs fan plots (mean/median/std/mode or 2D histograms).  
- `plot_moho_crustvel_tradeoff`: scatter of Moho depth vs crustal velocity for understanding trade-offs.  
- `plot_currentmodels`, `plot_currentdatafits`, `plot_bestmodels`, `plot_bestdatafits`: overlay observed data and best-fitting model responses.  
- `plot_rfcorr`: visualise assumed RF correlation factors.  
- `merge_pdfs`: compositing saved PDF plots into a single report.  
- `save_chainplots`, `save_plots`: batch-generate figures (per chain or global) for quick reporting.

#### `BayHunter.SynthObs.SynthObs`
- `return_swddata(h, vs, vpvs=1.73, pars=dict(), x=None)`: runs Surf96-based forward modeling for Rayleigh/Love dispersion curves across provided periods.  
- `return_rfdata(h, vs, vpvs=1.73, pars=dict(), x=None)`: runs RFmini for P- and S-receiver functions with tunable Gaussian filters, water level, slowness.  
- `save_data(data, outfile=None)` / `save_model(h, vs, vpvs, outfile=None)`: write synthetic observations or models to disk.  
- `compute_expnoise` / `compute_gaussnoise`: draw correlated noise vectors with exponential or Gaussian correlation laws.  
- `_nocorr`, `_gausscorr`, `_expcorr`: analytic covariance (inverse & log det) helpers matching the Valuation routines.  
- `compute_explike(yobss, ymods, noise, gauss, rcond=None)`: evaluate expected log-likelihoods for synthetic experiment design (useful to calibrate BayWatch reference values).

#### `BayHunter.utils`
- `SerializingSocket.send_array` / `recv_array`: ZeroMQ helper to send numpy arrays with metadata (dtype/shape).  
- `string_decode(section)`: parse ConfigObj sections, eval stringified tuples/lists.  
- `load_params(initfile)`: read defaults into `modelpriors` and `initparams`.  
- `load_params_user(initfile, station, slowness=7)`: translate user-specific paths, deduce slowness from RF files, and prepare station-specific prior/param dictionaries.  
- `save_baywatch_config`, `save_config`: pickle all necessary inversion settings/targets for playback and plotting.  
- `read_config`, `get_path`: convenience wrappers to load configs or locate default files.  
- `_compute_gaussnoise`, `compute_spectrum`, `gauss_fct`, `_min_fct`, `_spec_resample`: signal-processing utilities used to estimate RF correlation parameters.  
- `plot_rrf_estimate`, `rrf_estimate`: Monte-Carlo estimation and plotting of RF correlation factors from spectra (helps pick priors for `noise_corr`).

#### Forward model wrappers
- `BayHunter.rfmini_modrf.RFminiModRF`: wraps Joachim Saul’s RFmini code. Methods: `_init_obsparams` (deduce sampling/taper), `write_startmodel` (export models), `set_modelparams` (update Gauss/water/slowness), `compute_rf` (call Fortran backend to compute RF), `run_model` (type conversion + `compute_rf`).  
- `BayHunter.surf96_modsw.SurfDisp`: wraps Surf96. Methods: `set_modelparams`, `get_surftags` (map reference to Love/Rayleigh and phase/group types), `get_modelvectors` (pad to Surf96’s 100-layer arrays), `run_model` (call the compiled solver, handle interpolation if >60 periods).

### Dataset-generation subpackage (`BayHunter.data`)

#### `model.py`
- `SeismicParams`: stores inversion settings; methods `to_dict`, `from_dict`, `__str__`.  
- `SeismicPrior`: posterior-friendly representation of Vs, depth, layer count, vp/vs, Moho, mantle, and noise priors; methods mirror `SeismicParams`.  
- `DispersionCurve`: handles storing, plotting, serialising, and computing misfits between dispersion curves.  
- `SeismicModel`: convenient wrapper around the Voronoi representation with `model`, `vs`, `z`, `vpvs`, `nlayers` properties; `split_params`, `get_vp`, `get_vp_vs_h`, `forward` (calls the new `SurfDispModel` with user-chosen period grids), `plot`, `to_dict`/`save_*`, `_calc_synth`, and class methods `from_dict`, `load_*`.  
- `SeismicSample`: couples a `SeismicModel` with a `DispersionCurve` and exports Arrow-compatible dicts.  
- `SeismicSampleBatch`: accumulates samples and converts them to Arrow tables.  
- `SurfDispModel`: standalone forward wrapper for dataset generation with dynamic period grids (`__init__`, property accessors, `set_modelparams`, `run`, and static `get_modelvectors` for Surf96 input arrays).

#### `dataset.py`
- `save_dataset_info(...)`: writes metadata JSON describing priors, sampling settings, dispersion length, and generation command.  
- `generate_folds_json(...)`: splits shard file names into 2-/k-fold dictionaries for benchmarking splits.

#### `sample.py`
- `sample_vpvs`: draw a vp/vs ratio from either a scalar or interval prior.  
- `sample_noise`: sample per-target noise correlation and amplitude from prior ranges.  
- `sample_seismic_model`: draw Vs and depth Voronoi nuclei, apply Moho constraints, and loop until `validate_model` confirms compliance.  
- `vornoi_to_layers`: convert Voronoi parameterisation into a regular layered profile (`vs(z)`) for grid-based ML tasks.  
- `sample_model`: convenience wrapper around `sample_vpvs` + `sample_seismic_model`.  
- `forward_modeling`: placeholder hook for future workflows.

#### `validation.py`
- `validate_vpvs`: checks whether the vp/vs ratio lies inside the prior interval.  
- `validate_model`: enforces the same constraints as `_validmodel` (layer count, `thickmin`, vs/z bounds, LVZ/HVZ) but for dataset generation.

#### `utils.py`
- `save_samples_to_arrow(samples, output_path)`: convert a list of `SeismicSample` objects to a PyArrow table and write a compressed parquet file.

#### `huggingface.py`
- `generate_dataset_card(...)`: compose a README-style dataset card with metadata, JSON example, and CLI instructions.  
- `upload_dataset_to_hf(...)`: push local dataset folders to the HuggingFace Hub (optional automation).

#### `plotting.py`
- `plot_models(models, labels=None, colors=None, invert_axes=False, title="Seismic Models", show=True)`: overlay the `.plot()` output of multiple `SeismicModel` instances in one figure.

### CLI and utility functions (`BayHunter.__main__`)
- `cli()`: click command group entry point.  
- `tuple_of_ints(value)`: parse comma-separated integers.  
- `FloatListParamType`: custom Click type parsing comma-separated floats.  
- `safe_forward(model, length)`: run `SeismicModel.forward` in a thread with timeout protection.  
- `plot_2d_perturbations(...)`: perturb Vs and depth simultaneously, run forward models, plot heatmaps of misfit vs noise levels plus the base model.  
- `run_perturbations(...)`: helper for one-dimensional noise sweeps (either Vs or depth).  
- `plot-z-perturbations` / `plot-vs-perturbations` commands: wrappers calling `run_perturbations` for depth or velocity noise.  
- `noisy-forward`: generate multiple noisy perturbations of a base model, run forward modelling, and plot the ensemble of Vs profiles and dispersion curves.  
- `run-forward`: forward model a user-specified Vs-depth model; optional plotting and saving of dispersion curves/models.  
- `generate-dataset`: end-to-end dataset generator (reads `.ini` priors via `configparser`, draws samples, runs forward modelling with optional variable period grids, writes Parquet shards, metadata, folds, and dataset cards).  
- `forward-modeling`: either load a saved model or sample from priors, then run forward modelling with optional plots and output files.  
- `sample-model`: sample a single model from priors, print/plot/save it.  
- `surfdisp96`: thin wrapper that forwards user-specified arrays directly into the `SurfDispModel.run` solver and prints the dispersion curve in a table.  
- `generate`: quick random sampler writing JSON-ready dictionaries (currently prints summary only).

### Monitoring helpers outside classes
- `BayHunter.Plotting.vs_round(vs)`: round velocities to 0.025 km/s bins for histograms.  
- `BayHunter.Plotting.tryexcept`: decorator that wraps plotting methods to catch exceptions without stopping batch scripts.

## Next steps
- Use `baywatch.py` to monitor live runs once `MCMC_Optimizer.mp_inversion` is launched.  
- Leverage `PlotFromStorage.save_plots` after runs to generate posterior reports.  
- For benchmarking or data-driven workflows, rely on the `BayHunter.data` API plus the CLI commands under `python -m BayHunter`.

Armed with this overview and the per-class method reference, you can traverse the repository knowing where each physical or algorithmic concept is implemented, how proposals are generated and assessed, and how to reproduce or extend the MCMC inversion workflow.
