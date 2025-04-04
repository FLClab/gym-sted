# gym-sted

OpenAI gym implementation of the [pySTED simulation tool](https://github.com/FLClab/pySTED).

## Installation

### Gym

We recommend using a dedicated `python` environment.
```bash
conda create -n gym-sted python=3.10
conda activate gym-sted
```

Then you can install `gym-sted`.
```bash
pip install -e gym-sted
```

The experiments and models are available at this [repository](https://github.com/FLClab/gym-sted-pfrl).

## Experiments

`gym-sted` is a stand-alone librairy implemented under the OpenAI standards. The users may use the environment in their own code. However, we provide a detailed explanation on how to run experiments with the `gym-sted` environment in [gym-sted-pfrl](https://github.com/FLClab/gym-sted-pfrl).

## Environments

Many environments are available in `gym-sted`. The user may choose the environment that fits their needs. In the following, we provide a brief description of the environments. All environments are implemented in the `gym_sted.envs` module.

*Note. The environements that are not described below are in an experimental state. No guarantees are provided for these.*

### `ContextualMOSTED-easy-v0`

This environment is designed to train a model that can handle a specific fluorophore. The task of this environment is to detect nanoclusters in simulated spines. The reward of the agent corresponds to the F1 score of the acquisition.

This experiment was used in Turcotte et al. (2021).

### `MOSTEDRankingWithExpertDemonstrationsF1Score-easy-v0`

This environment is designed to train a model that can handle a specific fluorophore. The task of this environment is to detect nanoclusters in simulated spines. The reward of the agent corresponds to the ranking of the measured F1 score compared to a bank of expert demonstrations. While the F1 score is used to train the model, the agent only sees the reward as a ranking which is closer to real experimental condition.

This experiment was used in Turcotte et al. (2021).

### `ContexutalMOSTED-<easy/hard>-<condition>-v0`

These environments are designed to train a model that can handle a specific fluorophore in the `easy` mode or a range of fluorophore `hard`. The user may choose the fluorophore by setting the `<condition>`. The condition is only required when `easy` mode is selected. The conditions are: `hslb`, `hshb`, `lslb`, `lshb`.

The task of this environment is to detect nanoclusters in simulated spines. The reward of the agent corresponds to the F1 score of the acquisition.

### `PreferenceCountRateMOSTED-hard-v0`

This environment aims at training a model that can handle a large variety of fluorophores. The environment is designed to be challenging for the model. The reward of the model is calculated from the `PrefNet` model. A negative reward is given when the acquisition of the model has a high count rate. 

### `AbberiorMOSTED-v0`

This environment is used only for the deployment of the agent in a real environment. This environment is not used for training a model. The user may use this environment to acquire images on the microscope.

### `AbberiorMOSTEDCountRate-v0`

This is similar to the `AbberiorMOSTED-v0` environment. However, the count rate is calculated and returned to the agent.

## Citation

If you use `gym-sted` please cite the following papers

```bibtex
@article{bilodeau2024development,
  title={Development of AI-assisted microscopy frameworks through realistic simulation with pySTED},
  author={Bilodeau, Anthony and Michaud-Gagnon, Albert and Chabbert, Julia and Turcotte, Benoit and Heine, J{\"o}rn and Durand, Audrey and Lavoie-Cardinal, Flavie},
  journal={Nature Machine Intelligence},
  volume={6},
  number={10},
  pages={1197--1215},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
