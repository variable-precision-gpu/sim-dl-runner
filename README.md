# GPGPU-Sim DL Runner

This program manages the execution of deep learning programs on GPGPU-Sim.

## Requirements
- [x] Compile programs
- [x] Apply a global sim config file to all programs
- [x] Setup sim environment
- [x] Train  
Params: weights, epochs, start epoch, end epoch
- [x] Infer  
Params: weights, logs (not implemented yet)
- [x] Clean up intermediate files (_app_cuda_version, _cuobjdump_list_ptx)

## Usage
### Making DL Program Compatible
#### 1. Program needs to be compilable with `make`
#### 2. Program needs to accept the following modes:
Train
```console
PROGRAM -train EPOCHS WEIGHTS_FILE
```
Train (Increment)
```console
PROGRAM -train-increment START_EPOCH END_EPOCH INPUT_WEIGHTS_FILE OUTPUT_WEIGHTS_FILE
```
Test
```console
PROGRAM -test WEIGHTS_FILE
```

### Configuring Script
#### Repository Configuration
- Program directory
- Sim config directory
- Sim directory

#### Script Configuration
- Initial VF32 significand width
- Program name (directory name)
- Executable name
- Stage configuration (TODO: add more details)

### Executing Script
```console
$ python3 runner.py
```
