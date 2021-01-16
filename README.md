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
