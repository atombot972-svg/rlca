# TQL: Scaling Q-Functions with Transformers by Preventing Attention Collapse

<img src="assets/scale.png" width="100%">


## Overview

TQL unlocks scaling of value functions with transformers by preventing attention collapse. 

This repository contains code for running the TQL algorithm on OGBench tasks

## Installation

To setup the environment, run the following once (or after changing `pyproject.toml`):

```bash
uv sync
```

To activate the environment, run:

```bash
source .venv/bin/activate
```

To test your environment, run:
```bash
python main.py
```

## Debugging

The following env vars may be required:

```bash
export MUJOCO_GL=egl
export DISPLAY=
export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export PATH="$SITE_PACKAGES/nvidia/cuda_nvcc/bin:$PATH"
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH:/usr/lib/nvidia"
export CFLAGS="-DGLEW_NO_GLU ${CFLAGS:-}"
export CXXFLAGS="-DGLEW_NO_GLU ${CXXFLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Note: You may also need to install mujoco and add $HOME/.mujoco/mujoco210/bin to your LD_LIBRARY_PATH and PATH
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:$HOME/.local/lib64"
# export PATH="$PATH:$HOME/.mujoco/mujoco210/bin"
```

To test if your JAX installation is working w/CUDA, see whether this command shows GPU or CPU devices:
```bash
python -c "import jax; print(jax.local_devices())"
```


## Running experiments

The `agents` folder contains the implementation of algorithms and default hyperparameters. Here are some example commands to run experiments:

## Running experiments

The `agents` folder contains the implementation of algorithms and default hyperparameters. Example commands:

**Critic model size 26M** (`hidden_dim=1024`):

```bash
python main.py --env_name=cube-double-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=1024 --agent.num_heads=32

python main.py --env_name=cube-triple-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=1024 --agent.num_heads=32 --agent.critic_lr=4e-4

python main.py --env_name=puzzle-3x3-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=1000 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=1024 --agent.num_heads=32

python main.py --env_name=puzzle-4x4-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=1000 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=1024 --agent.num_heads=32

python main.py --env_name=scene-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=1024 --agent.num_heads=32
```

**Critic model size 7M** (`hidden_dim=512`):

```bash
python main.py --env_name=cube-double-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.0, 3.0), (2.5, 2.5))" --agent.hidden_dim=512 --agent.num_heads=16

python main.py --env_name=cube-triple-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=512 --agent.num_heads=16

python main.py --env_name=puzzle-3x3-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=1000 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=512 --agent.num_heads=16

python main.py --env_name=puzzle-4x4-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=1000 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=512 --agent.num_heads=16

python main.py --env_name=scene-play-singletask-{task1,task2,task3,task4,task5}-v0 --agent=agents/tql.py --agent.alpha=300 --agent.attention_entropy_target="((3.5, 3.5), (3.0, 3.0))" --agent.hidden_dim=512 --agent.num_heads=16
```

Complete list of hyperparameters for each method are available in the corresponding agent config files under `agents/`.


## Acknowledgments

This codebase is adapted from [OGBench](https://github.com/seohongpark/ogbench) and [FQL](https://github.com/seohongpark/fql) implementations.
