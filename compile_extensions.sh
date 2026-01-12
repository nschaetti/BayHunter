#!/bin/bash

# Fortran fixed-line friendly flags (safe on most systems)
export FC=gfortran
export NPY_DISTUTILS_APPEND_FLAGS=1
export FFLAGS="-ffixed-line-length-none"

python3 -m numpy.f2py -c -m surfdisp96_ext BayHunter/extensions/surfdisp96.f --fcompiler=gnu95 only: surfdisp96 :