
import setuptools
from setuptools import setup

setup(
    name='gym_sted',
    version='0.0.1',
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "gym>=0.26",
        "scikit-learn",
        "scikit-image",
        "numpy",
        "scipy",
        "matplotlib",
        "pysted",
        "torch"
    ]  # And any other dependencies foo needs
)
