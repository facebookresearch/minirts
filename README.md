# Hierarchical Decision Making by Generating and Following Natural Language Instructions

This is the repo for paper [Hierarchical Decision Making by Generating
and Following Natural Language Instructions](https://arxiv.org/abs/1906.00744).

## Dependencies
We write our model and training code using PyTorch and its C++
interface. It is a known issue that some strange behaviors can happen
if the compiler used for compiling this repo is differnet from the
compiler used by the pre-built PyTorch due to the incompatibility
between different versions of gcc. Therefore we recommand to **build
PyTorch from scratch** before compiling this project.

We recommand to use `conda` and follow the instruction
[here](https://github.com/pytorch/pytorch#from-source) to compile and
install PyTorch from source first. Then install dependency for this
project:
```
conda install lua numpy tqdm
conda install -c conda-forge tensorboardx
```

## Get Started

### Clone repo
```
git clone ...
git submodule sync && git submodule update --init --recursive
```

### Download dataset & pretrained models
To downalod and unzip the original replays, processed json files, and
dataset, from the following command. Note that it will take a while
for the command to finish.
```
cd data
sh download.sh
```

To download some pretrained models used in the paper:
```
cd pretrained_models
sh download.sh
python update_path.py
```

### Visualize dataset

We build a visualization tool that works
directly with json file so that people can get a more intuitive view
of the dataset and start working on it without compiling the game.
Please go to the visual folder for detailed instructions on how to use
it.

### Train models

We put the shell scripts that can be used to re-train
the model with configurations used in the paper in
`scripts/behavior_clone/scripts`. Simply run command like
```
sh scripts/coach_rnn500.sh
```
to start training. The command needs to be run under `behavior_clone`
folder. Normally it will take quite a while to load the dataset. For
quick testing and debugging, one can add `--dev` at the end of the
shell script to use the dev dataset instead, which contains only 2000
entries and thus much faster to load.

### Run matches between models

To run matches between trained models,
we first need to compile the game.  Please see the "Build" and "Set env
var" section for details. After the game is compiled, the following
command can be used to launch matches between an `RNN coach + RNN
executor` and `zero executor` (the one that does not use latent
language).

```
python match2.py --coach1 rnn500 --executor1 rnn \
	--coach2 rnn500 --executor2 zero \
	--num_thread 500 --seed 9999
```

## Structure

### scripts

This is the main folder for our algorithm, containing code for data
processing, model definition & training, and evaluation. See the
readme file for each subfolder for more details.

### visual

This contains a web tool for visualizing dataset from json so that we
can have a peek of the dataset without compiling the game.

### game

This folder contains the implementation of the game, including game
logic, some built-in AIs used for collecting data, as well as
necessary backends to extract features from game state for model
evaluation.

### tube

This folder defines a set of infra that dynamically batches data from
various C++ game threads and transfer them between C++ and Python.

## Build
```
mkdir build
cd build
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
cmake ..
make
```

## Set Env Variables

 Note that we need to set the following before
running any multi-threading program that uses the C++
torch::Tensor. Otherwise a simple tensor operation will use all cores
by default.
```
export OMP_NUM_THREADS=1
```
