
from setuptools import setup

setup(
    name='gym_sted',
    version='0.0.1',
    install_requires=[
        "gym>=0.26",
        "scikit-learn",
        "scikit-image",
        "numpy",
        "scipy",
        "matplotlib",
        "pysted @ git+https://github.com/FLClab/pySTED.git",
        "torch"
    ]  # And any other dependencies foo needs
)
