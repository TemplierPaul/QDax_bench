# QDax Benchmark

**This repository is still under development (WIP).**

The goal of QDax_bench is to provide the infrastructure to easily benchmark Quality-Diversity algorithms on a variety of tasks, using [QDax](github.com/adaptive-intelligent-robotics/QDax).  

New algorithms and tasks can be easily added using [Hydra](https://hydra.cc/docs/intro/).

Algorithms are built using [factories](qdax_bench/factories), and tasks are built using [setup functions](qdax_bench/utils/setup.py).


## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python main.py algo=me task=kh_standard
```  

## Features

### Tasks

- **QD optim**: Standard Black-Box optimization tasks, with QD descriptors using random projections ([Source](https://github.com/maxencefaldor/learned-qd/blob/main/learned_qd/tasks/bbob_fn.py))
- **Arm**: A simple 2D arm task
- **Kheperax**: 2D mazes using [Kheperax](https://github.com/adaptive-intelligent-robotics/Kheperax)
- **Brax**: Robotics tasks using [Brax](https://github.com/google/brax)

### Algorithms

- **ME**: MAP-Elites 
- **PGA-ME**: Policy-Gradient Assisted MAP-Elites 
- **DNS**: Dominated Novelty Search 
- **Extract-ME**: Extract MAP-Elites
- **Extract-DNS**: Extract ME with a DNS archive
