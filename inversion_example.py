

from rich import print
from rich.traceback import install

install(show_locals=True)

import logging
import numpy as np
import os.path as op
from BayHunter import utils
from BayHunter import SynthObs
from BayHunter import Targets
from BayHunter import MCMC_Optimizer
from BayHunter import PlotFromStorage


# Set global logging level to DEBUG
# logging.basicConfig(level=logging.DEBUG)

try:
    seed = 42
    np.random.seed(seed)

    print("[bold cyan]Loading synthetic observed data...[/bold cyan]")
    xsw, _ysw = np.loadtxt('tutorial/observed/st3_rdispph.dat').T
    xrf, _yrf = np.loadtxt('tutorial/observed/st3_prf.dat').T

    print("[bold cyan]Adding synthetic noise...[/bold cyan]")
    ysw = _ysw + SynthObs.compute_expnoise(_ysw, corr=0, sigma=0.012)
    yrf = _yrf + SynthObs.compute_gaussnoise(_yrf, corr=0.98, sigma=0.005)

    print("[bold cyan]Initializing targets...[/bold cyan]")
    target1 = Targets.RayleighDispersionPhase(xsw, ysw)
    target2 = Targets.PReceiverFunction(xrf, yrf)
    target2.moddata.plugin.set_modelparams(gauss=1.0, water=0.01, p=6.4)

    targets = Targets.JointTarget(targets=[target1, target2])

    print("[bold cyan]Loading parameters from config.ini...[/bold cyan]")
    priors, initparams = utils.load_params('tutorial/config.ini')

    print("[bold cyan]Saving config file for BayWatch...[/bold cyan]")
    utils.save_baywatch_config(targets, priors=priors, initparams=initparams)

    print("[bold cyan]Starting MCMC inversion...[/bold cyan]")
    optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors, random_seed=seed)
    optimizer.mp_inversion(nthreads=8, baywatch=True, dtsend=1)

    print("[bold cyan]Initializing plot object...[/bold cyan]")
    path = initparams['savepath']
    cfile = f"{initparams['station']}_config.pkl"
    configfile = op.join(path, 'data', cfile)
    obj = PlotFromStorage(configfile)

    print("[bold cyan]Saving posterior distributions and plots...[/bold cyan]")
    obj.save_final_distribution(maxmodels=100000, dev=0.05)
    obj.save_plots()
    obj.merge_pdfs()

    print("[bold green]Inversion completed and results saved successfully.[/bold green]")
except Exception as e:
    print(f"[bold red]An error occurred: {e}[/bold red]")
    raise
# end try

