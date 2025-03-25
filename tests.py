
import numpy as np
from BayHunter.surfdisp96_ext import surfdisp96


def call_fortran_surfdisp(h, vp, vs, rho, t, iwave=2, mode=1):
    """
    Call the Fortran surfdisp96 routine.
    """
    nlayer = len(h)
    kmax = len(t)
    cg = np.zero_like(t)
    err = np.zeros(1)

    # Call the Fortran routine
    surfdisp96_ext.surfdisp96(
        h.astype(np.float64),
        vp.astype(np.float64),
        vs.astype(np.float64),
        rho.astype(np.float64),
        int(nlayer),
        1,
        iwave,
        mode,
        1,
        int(kmax),
        t.astype(np.float64),
        cg,
        err
    )

    return cg, err[0]
# end call_fortran_surfdisp




