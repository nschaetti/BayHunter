#
#
#

import numpy as np


# Placeholder function for 'sphere' transformation:
def sphere(ifunc, flag, d, a, b, rho, mmax, llw):
    """
    Placeholder for Earth-flattening transformation (spherical earth correction).

    Parameters:
    ----------
    ifunc : int
        Function type indicator.
    flag : int
        Control flag (0 or 1) indicating preparation or execution.
    d, a, b, rho : ndarray
        Model layer properties (thickness, velocities, densities).
    mmax : int
        Number of layers.
    llw : int
        Water-layer indicator (1 no water, 2 water at surface).

    Notes:
    ------
    Detailed implementation of this routine depends on Biswas (1972) method or similar.
    """
    pass  # To be implemented
# end sphere


def gtsolh():
    pass
# end getsol


def getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst, d, a, b):
    return 0, 0
# end getsol


# Compte dispersion velocities for surface waves
def surfdisp2k25(
        thkm,
        vpm,
        vsm,
        rhom,
        nlayer,
        iflsph,
        iwave,
        mode,
        igr,
        kmax,
        t: np.ndarray
):
    """
    Computes seismic surface wave dispersion (phase/group velocities).

    Args:
        thkm (array): Layer thicknesses (km).
        vpm (array): P-wave velocities (km/s).
        vsm (array): S-wave velocities (km/s).
        rhom (array): Layer densities (g/cm³).
        nlayer (int): Number of layers.
        iflsph (int): Earth model type (0=flat, 1=spherical).
        iwave (int): Wave type (1=Love, 2=Rayleigh).
        mode (int): Mode number (1=fundamental, 2=first higher mode, ...).
        igr (int): Velocity type (0=phase, >0=group).
        kmax (int): Number of periods.
        t (array): Periods for calculation (s).

    Returns:
        tuple:
            cg (array): Calculated dispersion velocities (km/s).
            err (float): Error status (0.0=no error, 1.0=error).
    """
    err = 0.0
    cg = np.zeros(kmax)

    nsph = iflsph
    mmax = nlayer

    # Copy input model
    b = np.array(vsm)
    a = np.array(vpm)
    d = np.array(thkm)
    rho = np.array(rhom)

    # Wave type selection
    if iwave == 1:
        idispl, idispr = kmax, 0
    elif iwave == 2:
        idispl, idispr = 0, kmax
    else:
        raise ValueError("Invalid iwave (must be 1 or 2)")
    # end if

    # Constants
    sone0 = 1.5
    ddc0 = 0.005
    h0 = 0.005
    one = 1.0e-2
    onea = sone0

    llw = 1 if b[0] > 0 else 2

    if nsph == 1:
        sphere(0, 0, d, a, b, rho, mmax, llw)
    # end if

    # Extremal velocities
    betmx = -1e20
    betmn = 1e20
    jmn, jsol = 0, 1
    for i in range(mmax):
        if b[i] > 0.01 and b[i] < betmn:
            betmn = b[i]
            jmn, jsol = i, 1
        elif b[i] <= 0.01 and a[i] < betmn:
            betmn = a[i]
            jmn, jsol = i, 0
        # end if
        betmx = max(betmx, b[i])
    # end for

    # Initial phase velocity estimation
    cc1 = betmn if jsol == 0 else gtsolh(a[jmn], b[jmn])
    cc1 *= 0.95
    cc1 *= 0.90
    cc = cc1
    dc = abs(ddc0)
    cm = cc

    c = np.zeros(kmax)
    cb = np.zeros(kmax)

    # Root finding stops at ift
    ift = 999

    # Loop over wave type (Love/Rayleigh)
    # ifunc = 1 is Love, ifunc = 2 is Rayleigh
    for ifunc in [1, 2]:
        if (ifunc == 1 and idispl == 0) or (ifunc == 2 and idispr == 0):
            continue
        # end if

        # nsph = 1: Earth is a sphere, nsph = 0: Earth is flat
        if nsph == 1:
            sphere(ifunc, 1, d, a, b, rho, mmax, llw)
        # end if

        # Go through different modes
        # 1 is fundamental, 2 is first higher mode, etc.
        for iq in range(1, mode + 1):
            # Compute dispersion velocities for each period
            for k in range(kmax):
                # Stop root finding if k >= ift
                if k >= ift:
                    break
                # end if

                # Get the period
                t1 = t[k]

                # Approximate group velocity
                # with numerical differentiation.
                # Add a small relative perturbation.
                if igr > 0:
                    t1a = t1 / (1 + h0)
                    t1b = t1 / (1 - h0)
                    t1 = t1a
                else:
                    t1a = t1
                # end if

                # Estimate initial bounds
                # c1: initial guess for phase velocity
                # clow: lower bound for phase velocity
                # ifirst: flag for first iteration

                # First iteration, fondamental mode
                if k == 0 and iq == 1:
                    c1, clow, ifirst = cc, cc, True
                # First iteration, higher modes
                elif k == 0 and iq > 1:
                    c1, clow, ifirst = c[0] + one*dc, c[0] + one*dc, True
                # Subsequent iterations
                elif k > 0 and iq > 1:
                    ifirst = False
                    clow, c1 = c[k] + one*dc, c[k-1]
                    if c1 < clow:
                        c1 = clow
                    # end if
                # k>0 and iq==1
                # Subsequent iterations, fundamental mode
                else:
                    ifirst = False
                    c1, clow = c[k-1] - onea*dc, cm
                # end if

                # Root finding (phase velocity)
                iret, c1 = getsol(t1, c1, clow, dc, cm, betmx, ifunc, ifirst, d, a, b)
                if iret == -1:
                    ift = k
                    err = 1.0
                    cg[k:] = 0.0
                    break
                # end if

                c[k] = c1

                if igr > 0:
                    ifirst = False
                    clow, c2 = cb[k] + one*dc, c1 - onea*dc
                    iret, c2 = getsol(t1b, c2, clow, dc, cm, betmx, ifunc, ifirst, d, a, b)
                    if iret == -1:
                        c2 = c[k]
                    cb[k] = c2
                    gvel = (1/t1a - 1/t1b) / (1/(t1a*c1) - 1/(t1b*c2))
                    cg[k] = gvel
                else:
                    cg[k] = c[k]
                # end if
            # end for k
        # end for iq
    # end for ifunc

    return cg, err
# end surfdisp2k
