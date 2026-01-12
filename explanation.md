# BayHunter explained for complete beginners

This document translates the BayHunter technical notes (`bayhunter.md`) and the
source code into friendly language. You can read it even if you are new to
maths, programming, and geology. Every unfamiliar term is introduced with a
plain-language definition before BayHunter’s method is described.

---

## 1. What BayHunter tries to do

The speed of seismic waves (shakes created by earthquakes or instruments) tells
us what layers lie under a seismic station. BayHunter is a computer program
that guesses the underground structure **(a stack of layers with different
wave speeds)** from two types of observations:

1. **Surface wave dispersion curves** – how waves that skim the surface slow
   down or speed up with different periods (a period is the time for one full
   wave wiggle).
2. **Receiver functions** – how the incoming P-waves (faster compressional
   waves) convert into S-waves (slower shear waves) when they meet sharp
   underground contrasts like the crust–mantle boundary.

The structure is described by:

- `Vs` – shear-wave velocity (speed of S-waves) for each layer.
- `Vp` – compressional-wave velocity (speed of P-waves), tied to `Vs`
  through a simple ratio called `Vp/Vs`.
- `h` – thickness of each layer (how thick the slice of rock is).
- The depth of interfaces like the **Moho** (crust–mantle boundary) and the
  presence of low- or high-velocity zones.
- Noise parameters saying how uncertain each observed data point is and how
  one data point depends on another.

BayHunter finds many different underground models that could explain the
observations instead of forcing a single “best” answer. This allows the user to
see what is known with confidence and what remains uncertain.

---

## 2. Key concepts explained from scratch

| Term | Gentle explanation |
| --- | --- |
| **Layered model** | Think of a multi-layered cake. Each slice has its own thickness and flavor (here: wave speed). BayHunter calls each slice a **layer**. |
| **Voronoi nuclei** | Instead of fixing how many layers the cake has, BayHunter places “pegs” (nuclei) along depth. The layers are built around these pegs. Fewer pegs produce thicker layers; more pegs give thinner layers. This flexible description lives in `BayHunter/Model.py`. |
| **P-wave / S-wave** | Two wave types used in seismology. `P` stands for primary (fast compressional), `S` for secondary (slower shear). |
| **Surface waves** | Waves that stay near the Earth’s surface. Their speed depends on frequency/period, forming a **dispersion curve**. The `SurfDisp` class in `BayHunter/surf96_modsw.py` runs the SURF96 solver to compute the synthetic curves. |
| **Receiver functions (RFs)** | When P-waves dive down, meet a boundary, and convert into S-waves, we record the converted signal. The `RFminiModRF` class in `BayHunter/rfmini_modrf.py` wraps Joachim Saul’s RFmini program to simulate this process. |
| **Bayesian inversion** | A recipe for updating beliefs. We start with prior beliefs (`priors`), compare a trial model to the data via a likelihood function (`Valuation` in `BayHunter/Targets.py`), and end with posterior beliefs (what remains plausible after seeing the data). |
| **Prior** | “Before looking at the data, what models did we think were possible?” These ranges live in `defaults.ini` and can be overridden per run. |
| **Likelihood** | “If this model were true, how likely are the observed data?” It is computed as a weighted misfit using covariance matrices (accounts for measurement errors). |
| **Posterior** | “Given both prior beliefs and data, what's the probability of each model?” BayHunter samples from this distribution. |
| **Markov Chain Monte Carlo (MCMC)** | A random walk that explores complicated probability landscapes. Each “chain” proposes small changes, keeps good ones more often, but still tries bad ones to avoid getting stuck. Implemented in `BayHunter/SingleChain.py`. |
| **Transdimensional** | The program can change the number of layers while running (birth/death moves in `SingleChain`). This is powerful because we do not need to guess the complexity in advance. |
| **Noise correlation** | Real data are not independent points. If one measurement is high, neighbors might also be high. BayHunter models both the strength (`noise_sigma`) and the correlation length (`noise_corr`) per target. |
| **Joint target** | A bundle of individual data types (surface waves, receiver functions, or other future data). The `JointTarget` class combines them so the inversion honors all data simultaneously. |

---

## 3. Step-by-step BayHunter method

### 3.1 Preparing the ingredients

1. **Describe the priors (`BayHunter/utils.py`)**
   - Load `defaults.ini` (or a station-specific `.ini`) with `load_params` or
     `load_params_user`.
   - Priors include Vs limits, depth limits, number of layers, allowed Moho
     depths, vp/vs ratios, low-/high-velocity zone percentages, and noise
     ranges for each target type.
   - You can think of this as setting the playground boundaries so the sampler
     never builds impossible Earth models.

2. **Create the targets (`BayHunter/Targets.py`)**
   - Each `SingleTarget` wraps:
     - Observed data (`ObservedData`),
     - A forward model (`ModeledData`, backed by either `SurfDisp` or
       `RFminiModRF`),
     - A `Valuation` object that knows how to score mismatches.
   - `JointTarget` stitches several single targets together. When the sampler
     evaluates a model, it simultaneously compares it to all chosen data types.

3. **Set initial guesses (`SingleChain.draw_init*`)**
   - `draw_initmodel` samples Vs and depth pegs from the priors, optionally
     locking a pair of layers around the expected Moho depth.
   - `draw_initvpvs` samples the starting `Vp/Vs`.
   - `draw_initnoiseparams` picks starting noise correlation and amplitude for
     each target. Fixed values (single numbers in the priors) are respected.

### 3.2 Turning models into synthetic data

1. **Convert pegs to a layered model**
   - `Model.get_vp_vs_h` (in `BayHunter/Model.py`) converts the Voronoi
     representation into layer-by-layer thickness, Vs, and Vp values. Vp is
     derived by multiplying each Vs by the chosen `Vp/Vs` ratio.

2. **Run the forward solvers**
   - `SurfDisp.run_model` (SURF96) predicts how surface waves would travel
     through the proposed layers.
   - `RFminiModRF.run_model` computes the synthetic receiver functions.
   - These synthetic curves/data are stored inside each `SingleTarget`.

3. **Compute likelihoods with realistic noise**
   - For each target, `Valuation.get_covariance_*` builds a covariance matrix
     matching the assumed noise.
     - No correlation (`get_covariance_nocorr`): simplest, errors are
       independent.
     - Exponential correlation (`get_covariance_exp`): points close in period
       or time influence each other.
     - Gaussian correlation (`get_covariance_gauss`): used when correlation is
       fixed but non-zero, common for receiver functions.
   - The log-likelihood is

     \[
     \log L = -\frac{1}{2}(y - \hat{y})^T C^{-1} (y - \hat{y}) - \frac{1}{2}\log|C| - \frac{n}{2}\log 2\pi,
     \]

     where `y` are observations, `ŷ` are synthetic predictions, and `C` is the
     covariance. In plain words: we punish large, noise-unsafe mismatches.

### 3.3 Proposing and accepting new models

1. **Sampler basics (`SingleChain.iterate`)**
   - Each chain maintains a “current” model. To move forward, it randomly picks
     from several move types:
       - `vsmod`: tweak one layer’s Vs.
       - `zvmod`: move a depth peg up/down.
       - `birth`: create a new layer (add a peg).
       - `death`: remove a layer.
       - `vpvs`: nudge the global Vp/Vs ratio.
       - `noise`: adjust a noise hyper-parameter.
   - These moves are implemented via helper methods such as
     `_model_vschange`, `_model_layerbirth`, `_get_vpvs_proposal`, etc.

2. **Validity checks (`_validmodel`, `_validnoise`, `_validvpvs`)**
   - Before spending time on forward modelling, BayHunter ensures the proposal
     stays inside the priors (Vs limits, minimum thickness `thickmin`,
     low-/high-velocity zone percentages, vp/vs bounds, allowed noise ranges).
   - Invalid proposals are rejected immediately, saving computation.

3. **Likelihood comparison and acceptance**
   - Valid proposals call `JointTarget.evaluate` to get fresh log-likelihoods.
   - `SingleChain` then performs a Metropolis-Hastings decision: even if the
     new model is worse, it can be accepted with some probability to keep the
     exploration honest.
   - For fixed-dimensional moves (no change in layer count), the acceptance
     depends on the difference in log-likelihoods plus prior terms.
   - For birth/death moves, `get_acceptance_probability` adds Jacobian and
     proposal-density corrections following Bodin et al. (2012); this is what
     makes BayHunter **transdimensional**.

4. **Adaptive proposal widths (`adjust_propdist`)**
   - Every 1,000 iterations the sampler checks whether moves are accepted too
     often (steps too small) or too rarely (steps too big) and rescales the
     proposal standard deviations to stay within the desired acceptance window.

5. **Burn-in, main run, and weighting**
   - The user chooses `iter_burnin` (warm-up) and `iter_main` (production)
     lengths in `defaults.ini` or the config file.
   - When a proposal is rejected, BayHunter keeps the previous model but notes
     that it stayed “alive” for an extra iteration. Later,
     `get_weightedvalues` replicates models according to how long they were
     current, ensuring the posterior sample represents time spent in each state
     without storing every single iteration.
   - Results are saved with `save_finalmodels`, split into burn-in and posterior
     archives, optionally thinned to a `maxmodels` count to limit disk usage.

### 3.4 Combining many chains (`BayHunter/mcmcOptimizer.py`)

1. **Parallel execution**
   - `MCMC_Optimizer.mp_inversion` launches several `SingleChain` instances in
     parallel processes (up to the CPU count or the user’s `nthreads` choice).
   - Shared-memory numpy arrays (`_init_shareddata`) collect models, likelihoods,
     vp/vs ratios, and noise parameters across chains so that plotting and
     monitoring do not need to reload from disk.

2. **Monitoring**
   - A helper `monitor_process` can stream the latest accepted values to
     BayWatch through ZeroMQ sockets. This live feed enables real-time plots of
     data fits, vp/vs evolution, likelihood trends, etc.

3. **Post-run outputs**
   - Chains are thinned/weighted individually and saved.
   - Optional managers can return the in-memory chain objects to interactive
     notebooks for custom analysis.

### 3.5 Visualisation and quality control

1. **BayWatch (`BayHunter/BayWatch/BayWatcher.py`)**
   - Listens to the ZeroMQ stream and shows scrolling plots: Vs profiles,
     vp/vs time series, likelihood evolution, noise parameters, and misfit
     comparisons between observed and synthetic data.

2. **Static plotting (`BayHunter/Plotting/PlotFromStorage`)**
   - After the run, you can load the saved `.npy` archives and create
     comprehensive reports: histograms of vp/vs, noise parameters, number of
     layers, best-fitting models, 1D/2D Vs distributions, Moho depth versus
     crustal velocity trade-offs, and more.
   - Helper functions like `plot_currentmodels`, `plot_posterior_models2d`,
     `plot_posterior_noise`, and `merge_pdfs` provide ready-made figures.

3. **Synthetic experiments (`BayHunter/SynthObs/SynthObs.py`)**
   - Useful for training or validation: generate synthetic surface wave or
     receiver function data, inject controlled noise, and estimate the expected
     log-likelihood for a planned survey.

### 3.6 Dataset generation helpers (`BayHunter/data`)

For machine-learning or benchmarking tasks, BayHunter ships a subpackage that
can generate labelled datasets of dispersion curves and models.

- `dataset.py` defines object-style wrappers (`SeismicParams`, `SeismicModel`)
  and functions to save metadata (`save_dataset_info`) or create K-fold splits
  (`generate_folds_json`).
- `sample.py` implements the same priors and validators as the inversion:
  `sample_model` draws Vs-depth models, `sample_noise` returns noise settings,
  and `vornoi_to_layers` converts into a regular grid.
- `validation.py` enforces the physical rules outside the inversion context.
- `utils.py` and `huggingface.py` offer convenience routines for exporting
  samples to Arrow/Parquet files and uploading datasets to the HuggingFace Hub.
- The CLI defined in `BayHunter/__main__.py` exposes commands such as
  `python -m BayHunter generate-dataset` or `python -m BayHunter run-forward`
  for scripting these workflows without touching the Python API.

---

## 4. Putting it all together – a narrative workflow

1. **Set the stage**
   - Edit a configuration (based on `defaults.ini`) describing your station:
     Vs bounds, depth range, expected Moho depth, periods for dispersion data,
     number of iterations, etc.

2. **Read or import your observations**
   - Dispersion curves (`period` vs `phase velocity` or `group velocity`).
   - Receiver functions (time series representing P-to-S conversions).
   - Provide measurement errors. If you do not know them, BayHunter can learn
     reasonable amplitudes and correlations.

3. **Create targets**
   - Use the `BayHunter.Targets` API or the helper scripts (e.g.,
     `tutorial/tutorialhunt.py`) to bundle the observations, forward-model
     settings, and noise references into a `JointTarget`.

4. **Launch the inversion**
   - Instantiate `MCMC_Optimizer` with the joint target and chosen settings.
   - Call `mp_inversion()` to spawn the chains. Each chain will:
     1. Draw a random model from the priors.
     2. Iteratively propose new models (birth/death, Vs, depth, noise, vp/vs
        moves).
     3. Run forward models (`SurfDisp`, `RFmini`) when proposals pass the quick
        validity filters.
     4. Accept or reject proposals based on the Bayesian likelihood.
     5. Save weighted samples to shared arrays and, later, `.npy` files.

5. **Watch and control**
   - Use BayWatch to see live plots. Stop the run early if chains have clearly
     converged or adjust settings and restart if the exploration looks poor.

6. **Inspect the results**
   - Load the saved models with `PlotFromStorage`.
   - Look at depth–Vs fan plots, histograms, posterior distributions of vp/vs,
     noise parameters, and misfits.
   - Interpret the ensemble: e.g., the narrow spread of Moho depths indicates
     strong constraint; a broad distribution signals uncertainty.

7. **Optional: generate synthetic datasets**
   - Run `python -m BayHunter generate-dataset` to sample hundreds or thousands
     of models and dispersion curves for benchmarking algorithms or training
     neural networks.

---

## 5. Glossary of recurring terms

- **Acceptance ratio** – fraction of proposed moves that become the new
  current model. BayHunter adapts proposal widths to keep this ratio in a
  healthy range.
- **Birth/death move** – adds or removes a Voronoi nucleus (layer) during the
  MCMC run.
- **Covariance matrix** – a grid that stores how uncertain each data point is
  and how two data points co-vary. Needed to fairly weight misfits.
- **Forward model** – a simulator that predicts what the data would look like
  for a given Earth model.
- **Likelihood** – the probability of observing the data if a trial model were
  true; higher values mean better fits.
- **Mahalanobis misfit** – formal name for the noise-aware mismatch BayHunter
  uses; essentially a weighted squared difference.
- **Moho** – shorthand for the Mohorovičić discontinuity, the transition from
  crust to mantle.
- **Noise hyper-parameters** – extra knobs (amplitude and correlation length)
  describing measurement uncertainties per data type.
- **Posterior sample** – the list of models kept after burn-in, representing
  the Bayesian answer.
- **Prior** – the bounds and preferences defined before data are considered.
- **Surface-wave dispersion** – a curve showing how the speed of Rayleigh or
  Love waves varies with wave period.
- **Vp/Vs ratio** – a coarse indicator of rock type; used to convert Vs to Vp.
- **Weighting** – BayHunter counts how many iterations a model stayed current
  to avoid storing duplicate entries explicitly.

---

## 6. Where to look in the code

| Capability | Main files / classes |
| --- | --- |
| Parameter sampling & validation | `BayHunter/SingleChain.py`, `BayHunter/Model.py`, `BayHunter/utils.py` |
| Targets & likelihoods | `BayHunter/Targets.py`, `BayHunter/rfmini_modrf.py`, `BayHunter/surf96_modsw.py` |
| Noise handling | `BayHunter/Targets.py` (`Valuation` and covariance helpers), `SingleChain.draw_initnoiseparams` |
| Sampler moves & acceptance | `BayHunter/SingleChain.py` (`iterate`, `_model_*` helpers, `get_acceptance_probability`) |
| Multi-chain orchestration | `BayHunter/mcmcOptimizer.py`, `batch_bayhunter.zsh` (wrapper script) |
| Live monitoring | `BayHunter/BayWatch`, `BayHunter/utils.SerializingSocket` |
| Plotting & reporting | `BayHunter/Plotting`, `tutorial/` examples |
| Synthetic datasets | `BayHunter/data` subpackage, `python -m BayHunter` CLI |

Keep this map handy while browsing the repository; it connects the plain words
above with the exact source locations for deeper study.

---

## 7. Final remarks for newcomers

- You do **not** need to understand every mathematical detail to run BayHunter.
  Focus on setting sensible priors and checking whether the posterior ensemble
  makes geologic sense.
- Always inspect the diversity of accepted models rather than only the very
  best-fitting one. The spread reveals uncertainty and possible trade-offs
  (e.g., a thicker crust can trade off with lower velocities).
- Practice on synthetic data (generated with `BayHunter.SynthObs`) before
  inverting real observations—this builds intuition about how priors and noise
  influence the outcome.
- If the sampler struggles (very low acceptance, stuck chains), adjust the
  `propdist` entries, widen priors, or revisit data error estimates.

By combining careful priors, physically informed forward models, and a robust
transdimensional MCMC sampler, BayHunter delivers a transparent, probabilistic
picture of the subsurface that even non-experts can interpret with the help of
this beginner-friendly guide.
