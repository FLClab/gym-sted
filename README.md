# gym-sted

OpenAI gym implementation of the pysted simulator

## Installation

This requires to install `pysted`
```bash
git clone https://github.com/bturc/audurand_pysted.git
pip install -e audurand_pysted
```

This also requires to install `baselines`. I installed tensorflow 2.X, this implies that I had to install `baselines` from the `tf2` branch
```bash
git clone https://github.com/openai/baselines.git
git checkout origin/tf2
pip install -e baselines
```

Then you can install gym-sted. I am currently forcing `gym==0.14` since this is the only compatible version of `gym` which worked for me.
```bash
pip install -e .
```

## Running an experiment

We will use the `baselines` library to run the experiments.
```bash
python -m baselines.run --alg=ppo2 --env=gym_sted:STED-v0 --network=cnn --num_timesteps=0
```
