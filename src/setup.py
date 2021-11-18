from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Egypt ABM Model',
    ext_modules=cythonize('./src/CythonFunctions.pyx'),
    zip_safe=False
)