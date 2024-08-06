from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(name="fast_binning",
              sources=["fast_binning.pyx",
                       ],
              include_dirs=[numpy.get_include()],
              define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
              ),
]

setup(
    name="fast_binning",
    ext_modules=cythonize(ext_modules, annotate=True),
)
