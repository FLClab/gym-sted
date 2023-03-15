# gym-sted

OpenAI gym implementation of the pysted simulator

## Installation

### Gym

We recommend using a `Python` environment.
```bash
conda create -n gym-sted python=3.10
conda activate gym-sted
```

This requires to install `pysted` and `metrics`
```bash
pip install git+https://github.com/FLClab/pySTED
pip install git+https://github.com/FLClab/metrics
```

Then you can install `gym-sted`.
```bash
pip install -e gym-sted
```

The experiments and models are available at this [repository](https://github.com/FLClab/gym-sted-pfrl).
```bash
git clone https://github.com/FLClab/gym-sted-pfrl.git
pip install -e gym-sted-pfrl
```

## Experiments

`gym-sted` is a stand-alone librairy implemented under the OpenAI standards. The users may use the environment in their own code. However, we provide a detailed explanation on how to run experiments with the `gym-sted` environment in [gym-sted-pfrl](https://github.com/FLClab/gym-sted-pfrl#usage).

## Citation

If you use `gym-sted` please cite the following paper
```bibtex
@misc{turcotte2021pysted,
  title = {pySTED : A STED Microscopy Simulation Tool for Machine Learning Training},
  author = {Turcotte, Benoit and Bilodeau, Anthony and Lavoie-Cardinal, Flavie and Durand, Audrey},
  year = {2021},
  note = {Accepted to AAAI 2021}
}
```
