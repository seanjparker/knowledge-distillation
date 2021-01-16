# Knowledge Distillation
[Github Repository](https://github.com/seanjparker/knowledge-distillation)

## Requirements

- Python 3 (>=3.7)
- Pipenv
- CUDA-compatible GPU (for training)

## Setup
As long as you have Python3 and Pipenv installed, run the following command: 

```bash
$ pipenv install
```

Pipenv will automatically create and install all the required dependencies for the project.

## Getting started

The quickest way to train some of the models is to load one of the notebooks in Google Colab.
For example, running the notebook found in `notebooks/kd-mnist.ipynb` you can train both the Teacher and Student
models on the full MNIST dataset as well as performing KD from the Teacher to the Student.
The Notebook outputs the accuracy of the models.

After running the notebook, you could then load the created model into the notebook `notebook/kd-analysis.ipynb`
which allows you to generate graphs to see how temperature affects the logits of the models.

Otherwise, you can perform training using the provided `run.py` script which uses the code from the notebooks and `train.py`
to train the models from scratch and perform KD using a varitey of hyperparameters.

First, alter the `run.py` script to run your desired experiment, then:
```shell script
$ pipenv shell
$ python notebooks/run.py
```

Alternatively, you can launch Jupyter Lab locally (the package is included in the Pipfile).
```shell script
$ pipenv shell
$ jupyter lab
```

## Navigating the codebase

- `/models` contains the best trained models for the Teacher, Student and Assistant networks
- `/notebooks` contains all the code for the project, it is broken down as follows:
    - `kd-analysis` allows you to generate plots based on the pickled data produced during training
    - `kd-assistant` contains the code from the TAKD experiments
    - `kd-cifar` contains all the experiments for KD teacher to student using the whole dataset and only 3% of the data
    - `kd-mnist` same as above, except for it uses the MNIST dataset
    - `model.py` contains network definitions and helper create functions
    - `train.py` contains helper functions for training the networks
    - `utils.py` contains functions for loading datasets and calculating model accuracy
    - `run.py` contains helper functions for running different experiments

