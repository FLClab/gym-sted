# gym-sted

OpenAI gym implementation of the pysted simulator

## Installation

### Gym

We recommend using a `Python` environment.
```bash
conda create -n gym-sted python=3.8
conda activate gym-sted
```

This requires to install `pysted`
```bash
git clone https://github.com/bturc/audurand_pysted.git
pip install -e audurand_pysted
```

Then you can install `gym-sted`.
```bash
pip install -e .
```

### Baselines

The baseline models are available at this [repository](https://github.com/FLClab/gym-sted-pfrl).
```bash
git clone git@github.com:FLClab/gym-sted-pfrl.git
pip install -e gym-sted-pfrl
```

## Running an experiment

We use `gym-sted-pfrl` to run the experiment.
```bash
cd gym-sted-pfrl
python main.py --env gym_sted:STEDdebug-v0 --batchsize=16 --reward-scale-factor=1.0 --eval-interval=100 --eval-n-runs=5
```
