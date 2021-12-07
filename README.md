# gym-sted

OpenAI gym implementation of the pySTED simulator

## Description

Despite the advantages in resolution granted by super-resolution fluorescence microscopy, the techniques remain challenging to use for non-expert users due to the large number of objectives which need to be optimized to obtain high quality images. Artificial intelligence, in particular reinforcement learning, could prove useful in assisting or controlling image acquisition. However, reinforcement learning approaches are data-hungry in training, rendering their application to super-resolution microscopy infeasible due to the large amount of sample waste training would require. 

`gym-sted` is a environment implemented using OpenAI gym in which the agent must carefully select the imaging parameters of a super-resolution STED microscopy setup in order to achieve some imaging goals.

## Installation

We recommend using a Python environment. Our installation was tested using Python 3.7.
```bash
conda create -n gym-sted python=3.7
conda activate gym-sted
```

This requires to install `pysted`
```bash
git clone https://github.com/FLClab/pySTED.git
pip install -r pySTED/requirements.txt
pip install -e pySTED
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

`gym-sted` is a stand-alone librairy implemented under the OpenAI standards. The users may use the environment in their own code. However, we provide a detailed explanation on how to run experiments with the `gym-sted` environment in [gym-sted-pfrl](https://github.com/FLClab/gym-sted-pfrl).

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
