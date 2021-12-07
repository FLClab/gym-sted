# gym-sted

OpenAI gym implementation of the pysted simulator

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
