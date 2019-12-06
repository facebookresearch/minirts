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
```bash
conda install lua numpy tqdm
conda install -c conda-forge tensorboardx
```

## Get Started

### Clone repo
```bash
git clone ...
git submodule sync && git submodule update --init --recursive
```

### Download dataset & pretrained models
To downalod and unzip the original replays, processed json files, and
dataset, from the following command. Note that it will take a while
for the command to finish.

```bash
cd data
sh download.sh
```

To download some pretrained models used in the paper:
```bash
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

```bash
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

```bash
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
```bash
mkdir build
cd build
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
cmake ..
make
```

## Set Env Variables

Note that we need to set the following before running any
multi-threading program that uses the C++ torch::Tensor. Otherwise a
simple tensor operation will use all cores by default.
```bash
export OMP_NUM_THREADS=1
```

## Control Executor with Text
We can control the executor ourselves by inputting text command to the
trained executor.  First we need to set up the web server for the
backend so that we can watch the gameplay in browser while controlling
the executor.

We provide a script to install apache without root access. If you have
root privilege, you can simply run `sudo apt-get update & sudo apt-get
install apache2`
```bash
cd ROOT
sh install_apache.sh
```
After installation finishes,
edit `/private/home/hengyuan/minirts-release/apache/httpd/conf/httpd.conf`
to change the `Listen 80` (line52) to `Listen 8000` or any number >1024. The reason is
that the ports with lower numbers are reserved by system and requires sudo to use them.

Then we need to link our frontend code to the apache root directory & start server
```bash
cd ROOT
ln -s $PWD/game/frontend $PWD/apache/httpd/htdocs/game
./bin/apachectl start
```

Now open a browser and navigate to `http://localhost:8000/`. You should see `It Works`.
Otherwise there are some issue with the server set up.

Then we can start a human game!
```bash
cd ROOT/scripts/behavior_clone
python human_coach.py --resource 500 --verbose
# it should show 'Waiting for websocket client ...'
```
On the browser, navigate to
`http://localhost:8000/game/minirts.html?player_type=spectator&port=8002`
and wait for the model to be loaded. The command line will prompt the
top 500 instructions the model was trained on. If you are using RNN
executor (by default), you don't have to choose from these
instructions as the RNN can ideally handle unseen combinations. If you
are using OneHot executor, you should input an instruction from the
list.

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@article{DBLP:journals/corr/abs-1906-00744,
  author    = {Hengyuan Hu and
               Denis Yarats and
               Qucheng Gong and
               Yuandong Tian and
               Mike Lewis},
  title     = {Hierarchical Decision Making by Generating and Following Natural Language
               Instructions},
  journal   = {CoRR},
  volume    = {abs/1906.00744},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.00744},
  archivePrefix = {arXiv},
  eprint    = {1906.00744},
  timestamp = {Thu, 13 Jun 2019 13:36:00 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-00744},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Copyright
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
