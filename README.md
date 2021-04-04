# GPGPU-Sim Deep Learning Runner

This project helps manage the execution of deep learning programs on GPGPU-Sim.

## Prerequisites
Python 3.7, CUDA 8, GPGPU-Sim 4.0, MPFR 3.1

## Execution
```console
$ python3 runner.py
```
Modify the `runner.py` file to set execution parameters.

## Adding DL Applications
New programs should be made compatible:
1. Make program compilable with `make`
2. Write the program to run in the following modes:

**Train**
```console
PROGRAM -train EPOCHS WEIGHTS_FILE
```
**Train (Increment)**
```console
PROGRAM -train-increment START_EPOCH END_EPOCH INPUT_WEIGHTS_FILE OUTPUT_WEIGHTS_FILE
```
**Test**
```console
PROGRAM -test WEIGHTS_FILE
```

3. Generate and override PTX