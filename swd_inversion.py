
# Imports
import numpy as np
import os.path as op
from rich import print
from rich.traceback import install


# Activate Rich
install(show_locals=True)

# BayHunter imports
from BayHunter import utils
from BayHunter import SynthObs
from BayHunter import Targets
from BayHunter import MCMC_Optimizer
from BayHunter import PlotFromStorage

def main():
    print("[bold cyan]Loading synthetic data...[/bold cyan]")
    xsw, _ysw = np.loadtxt('tutorial/observed/st3_rdispph.dat').T

    print("[bold cyan]Adding exponentially correlated noise...[/bold cyan]")
    ysw = _ysw + SynthObs.compute_expnoise(data_obs=_ysw, corr=0, sigma=0.012)

    print("[bold cyan]Initializing RayleighDispersionPhase target...[/bold cyan]")
    target_swd = Targets.RayleighDispersionPhase(xsw, ysw)

    # Join the targets
    targets = Targets.JointTarget(targets=[target_swd])

    print("[bold cyan]Defining prior and init parameters...[/bold cyan]")
    priors = {
        'vpvs': (1.4, 2.1),
        'layers': (1, 20),
        'vs': (2, 5),
        'swdnoise_corr': 0.0,
        'swdnoise_std': (1e-5, 0.05)
    }

    initparams = {
        'nchains': 5,
        'iter_burnin': (2048 * 16),
        'iter_main': (2048 * 8),
        'propdist': (0.015, 0.015, 0.015, 0.005, 0.005),
        'acceptance': (40, 45),
        'thickmin': 0.1,
        'rcond': 1e-5,
        'station': 'migrate_synth',
        'savepath': 'results',
        'maxmodels': 50000
    }

    print("[bold cyan]Saving config for BayWatch...[/bold cyan]")
    utils.save_baywatch_config(
        targets=targets,
        priors=priors,
        initparams=initparams
    )

    print("[bold green]Starting MCMC inversion...[/bold green]")
    optimizer = MCMC_Optimizer(
        targets=targets,
        initparams=initparams,
        priors=priors,
        random_seed=None
    )

    optimizer.mp_inversion(
        nthreads=8,
        baywatch=True,
        dtsend=1
    )

    print("[bold cyan]Post-processing results...[/bold cyan]")
    path = initparams['savepath']
    cfile = f"{initparams['station']}_config.pkl"
    configfile = op.join(path, 'data', cfile)
    obj = PlotFromStorage(configfile=configfile)

    obj.save_final_distribution(maxmodels=100000, dev=0.05)
    obj.save_plots()
    obj.merge_pdfs()

    print("[bold green]✅ Done! Results saved in:[/bold green]", path)
# end main

if __name__ == "__main__":
    main()
# End of file