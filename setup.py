# For editable install
from setuptools import setup
from setuptools.extension import Extension
from setuptools.config import read_configuration


config = read_configuration('setup.cfg')

setup(
      # setup parameters, all put in setup.cfg
      install_requires=config['options']['install_requires'],
      #tests_require=config['options']['tests_require'],
      )

from Cython.Build import cythonize
ext_modules=[
    Extension('tbmalt.ml.cmbtr', ['tbmalt/ml/cmbtr.pyx']),
    Extension('tbmalt.ml.cfeature', ['tbmalt/ml/cfeature.pyx']),
    Extension('tbmalt.ml.cacsf_pair', ['tbmalt/ml/cacsf_pair.pyx'])
]

import numpy
setup(ext_modules=cythonize(ext_modules), include_dirs=[numpy.get_include()])

