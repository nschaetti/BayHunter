#!/usr/bin/env python
try:
    from numpy.distutils.core import Extension as NumpyExtension
    from numpy.distutils.core import setup

    from distutils.extension import Extension
    from Cython.Build import cythonize

    import numpy

except ImportError:
    raise ImportError('Numpy needs to be installed or updated.')


extensions = [
    NumpyExtension(
        name='BayHunter.surfdisp96_ext',
        sources=['BayHunter/extensions/surfdisp96.f'],
        extra_f77_compile_args='-O3 -ffixed-line-length-none -fbounds-check -m64'.split(),  # noqa
        f2py_options=['only:', 'surfdisp96', ':'],
        language='f77'
    ),
    NumpyExtension(
        name='BayHunter.sphere_ext',
        sources=['BayHunter/extensions/sphere96.f'],
        extra_f77_compile_args='-O3 -ffixed-line-length-none -fbounds-check -m64'.split(),  # noqa
        f2py_options=['only:', 'sphere', ':'],
        language='f77'
    )
]

extensions.extend(cythonize(
    Extension("BayHunter.rfmini", [
        "BayHunter/extensions/rfmini/rfmini.pyx",
        "BayHunter/extensions/rfmini/greens.cpp",
        "BayHunter/extensions/rfmini/model.cpp",
        "BayHunter/extensions/rfmini/pd.cpp",
        "BayHunter/extensions/rfmini/synrf.cpp",
        "BayHunter/extensions/rfmini/wrap.cpp",
        "BayHunter/extensions/rfmini/fork.cpp"],
        include_dirs=[numpy.get_include()])))


setup(
    name="BayHunter",
    version="2.1",
    author="Jennifer Dreiling",
    author_email="jennifer.dreiling@gfz-potsdam.de",
    description=("Transdimensional Bayesian Inversion of RF and/or SWD."),
    install_requires=[],
    url="https://github.com/jenndrei/BayHunter",
    packages=['BayHunter'],
    package_dir={
        'BayHunter': 'BayHunter'},

    scripts=['BayHunter/scripts/baywatch'],

    package_data={
        'BayHunter': ['defaults/*'], },

    ext_modules=extensions
)
