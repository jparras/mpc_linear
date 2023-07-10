# An efficient underwater navigation method using MPC with unknown kinematics and non-linear disturbances

## Introduction

Code used to obtain the results in the paper Barreno, P., Parras, J., & Zazo, S. (2023). An efficient underwater navigation method using MPC with unknown kinematics and non-linear disturbances. Journal of Marine Science and Engineering. 2023, 11, 710. [DOI](https://doi.org/10.3390/jmse11040710).

## Launch

This project has been tested on Python 3.7.0 on Ubuntu 20.04. To run this project, create a `virtualenv` (recomended) and then install the requirements as:

```
$ pip install -r requirements.txt
```

To show the results obtained in the paper, just run the given main:
```
$ python main.py
```

In case that you want to train or adjust anything, do the changes wished, ensure that the train flag is set to `True` in the `main.py` file, as well as the number of threads for training using joblib, and run the same order as before. Note that the results file will be overwritten. 
