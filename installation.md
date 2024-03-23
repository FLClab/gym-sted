# Installation

Create a python environment
```bash
conda create --name gym-sted python=3.10
conda activate gym-sted
```

Clone required repositories
```bash
git clone https://github.com/FLClab/gym-sted-dev.git
git clone https://github.com/FLClab/gym-sted-pfrl-dev.git
git clone https://github.com/FLClab/metrics.git
git clone https://github.com/FLClab/Abberior-STED.git
```

Install the required repositories
```bash
pip install -e gym-sted-dev
pip install -e gym-sted-pfrl-dev
pip install -e metrics
pip install -e Abberior-STED
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
pip install -e gym-sted-pfrl-dev --no-index
```
