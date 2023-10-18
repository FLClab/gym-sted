# Installation

## Required repositories

```bash
git clone https://github.com/FLClab/gym-sted-dev.git
git clone https://github.com/FLClab/gym-sted-pfrl.git
git clone https://github.com/FLClab/metrics.git
git clone https://github.com/FLClab/Abberior-STED.git
```

## Compute Canada

Create python environment
```bash
module load python/3.10
module load scipy-stack

virtualenv --no-download ~/venvs/gym-sted
source ~/venvs/gym-sted/bin/activate
pip install --no-index --upgrade pip
```

Install denpencies
```bash
pip install -e Abberior-STED --no-index
pip install -e metrics --no-index
pip install -e gym-sted-dev
pip install -e gym-sted-pfrl --no-index
```
