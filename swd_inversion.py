
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

    print("[bold cyan]Loading parameters from config.ini...[/bold cyan]")
    priors, initparams = utils.load_params('tutorial/config.ini')
    initparams['savepath'] = 'swd_results'

    print(f"[yellow]Priors: {priors}[/]")
    print(f"[yellow]Initparams: {initparams}[/]")

    print("[bold cyan]Saving config for BayWatch...[/bold cyan]")
    utils.save_baywatch_config(
        targets=targets,
        priors=priors,
        initparams=initparams
    )

    # MCMC inversion using the MCMC_Optimizer class
    optimizer = MCMC_Optimizer(
        targets=targets,
        initparams=initparams,
        priors=priors,
        random_seed=None
    )

    print("[bold green]Starting MCMC inversion...[/bold green]")
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

    # print("[bold cyan]Saving posterior distributions and plots...[/bold cyan]")
    # obj.save_final_distribution(maxmodels=100000, dev=0.05)
    # obj.save_plots()
    # obj.merge_pdfs()
    models = obj.get_models(['models'], final=True)
    print(models.shape)
    print(models[0])
    print("[bold green]✅ Done! Results saved in:[/bold green]", path)
# end main

if __name__ == "__main__":
    main()
# End of file