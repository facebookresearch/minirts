# Hierarchical Decision Making by Generating and Following Natural Language Instructions
## Supporting code

### Dependencies
We implement our models in PyTorch 1.0.0, thus the following packages need to be installed:
```
pip intall ...
```



### Dataset
First, download `dataset.tar.gz` using http://bit.ly/2VL9lg6 and put in the root folder of this project. Then unpack it by running: 
```
tar -xzf dataset.tar.gz
```
This will create the `dataset` folder, with the following files:
```
dict.pt    # pickle archive of behavior_clone/inst_dict.py, which contains the words and instructions dictionaries.
dev.json   # a json dump of a small training dataset useful for debugging.
train.json # a json dump of a large training dataset.
val.json   # a json dump of a validation dataset.
```


### Training Executor Model


### Training Instructor Model

### Play against Rule-Based AI
