from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy


ext_modules = [
    Extension("pyfm_fast",
              ["pyfm_fast.pyx"],
              libraries=["m"],
              include_dirs=[numpy.get_include()])
]


setup(maintainer='Moriaki Saigusa',
      name='pyfm',
      packages=find_packages(),
      url='https://github.com/moriaki3193/pyFM',
      cmdclass={'build_ext': build_ext},
      ext_modules=ext_modules,
      test_suite='test')
