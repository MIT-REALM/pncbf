<div align="center">

# Policy Neural Control Barrier Function (PNCBF)

</div>

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png" width="16%" />
    <img src="./media/teaser.gif" width="32%" />
    &ensp;
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png" width="16%" />
</p>

<div align="center">

### How to train your neural control barrier function: Learning safety filters for complex input-constrained systems
[Oswin So](oswinso.xyz), [Zachary Serlin](https://zacharyserlin.com), [Makai Mann](https://makaimann.github.io),
Jake Gonzales, Kwesi Rutledge, [Nicholas Roy](https://aeroastro.mit.edu/people/nicholas-roy), [Chuchu Fan](https://chuchu.mit.edu)

[Webpage](https://mit-realm.github.io/pncbf/) •
[arXiv](https://arxiv.org/abs/2310.15478) •
[Paper](https://arxiv.org/pdf/2310.15478) &ensp; ❘ &ensp;
[Installation](#installation) •
[Getting started](#getting-started) •
[Citation](#citation)

</div>

## Installation
This is a [JAX](https://github.com/google/jax)-based project. To install, install `jax` first and other prereqs following their [instructions](https://jax.readthedocs.io/en/latest/installation.html).
Note that the `jax` version used in this project is quite old (`0.4.28`).
Next, clone the repository and install the package.
```bash
git clone https://github.com/mit-realm/pncbf.git
cd pncbf
pip install -e .
```

## Getting started
Example on the double integrator:

```bash
python scripts/dbint/pncbf_dbint.py --name dbint
```

To eval,
```bash
python scripts/dbint/eval_pncbf_dbint.py runs/pncbf_dbint/path_to_run/ckpts/5000
```

## Citation
Please cite the [PNCBF paper](https://arxiv.org/abs/2310.15478).
```bibtex
@inproceedings{so2024train,
  title={How to train your neural control barrier function: Learning safety filters for complex input-constrained systems},
  author={So, Oswin and Serlin, Zachary and Mann, Makai and Gonzales, Jake and Rutledge, Kwesi and Roy, Nicholas and Fan, Chuchu},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={11532--11539},
  year={2024},
  organization={IEEE}
}
```
