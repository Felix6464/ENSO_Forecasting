"""setup.py for s2aenso."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='s2aenso',
    version='0.1',
    description=('Package for seasonal to annual forecasting of ENSO.'),
    author='Jakob Schlör',
    author_email='jakob.schloer@uni-tuebingen.de',
    packages=find_packages()
)
