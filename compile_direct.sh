#!/bin/bash

# Set the Fortran compiler
FC=gfortran

# Set compilation flags
# -shared: Create a shared object file
# -fPIC: Position Independent Code (required for shared libraries)
# -ffixed-line-length-none: Allow Fortran fixed-form lines of any length
FFLAGS="-shared -fPIC -ffixed-line-length-none"

# Compile the Fortran file into a shared object
$FC $FFLAGS BayHunter/extensions/surfdisp96.f -o BayHunter/extensions/surfdisp96_ext.so

echo "Compilation completed. Check for any errors above."