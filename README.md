# GPGPU-Sim Deep Learning Runner

This project serves to organize the files required for running mixed-precision deep learning on the simulator. In addition, the `runner.py` script configures and serves as the entry point for execution runs, automating the various steps involved.

```
├─ config/
│  ├─ config_volta_islip.icnt
│  ├─ gpgpusim.config
├─ programs/
│  ├─ CNN2/
│  ├─ MLP2/
├─ README.md
├─ runner.py
├─ utils.py
```

`config/`: Central location containing the simulator configuration files to be applied to all programs. Replace with other config files in the simulator repository to simulate other GPUs.

`programs/`: Directory for deep learning projects as git submodules. Making the projects submodules allows the repositories to be updated independently.

`runner.py`: Python 3 script automating the simulator set up and clean up. Refer to the inline documentation in the script for details about the configuration parameters.

`utils.py`: Python 3 utility functions. Of note, `mpfr_exponent_range()` computes the environment variables one should set for the VF32 type in order to emulate a specific exponent and significand width.

## Prerequisites
- Python:  >=3.7
- Modified GPGPU-Sim

## Usage

### 1. Configuring the Simulator
Place the configuration files of the GPU being simulated in the `configs/` folder, instead of directly in the deep learning program's directory.

The mixed precision experiments conducted so far were performed with the TITAN V configuration that's already in the folder, but feel free to experiment with other configurations. Note that some runtime issues were encountered previously with the TITAN X config.

### 2. Adding new Deep Learning Programs
Add a new deep learning application into the programs subdirectory as a git submodule

```console
$ git submodule add <repo_url>
```

All new programs added will need to be made compatible with the runner script. This can be done by modifying the program to conform to the following rules:

1\. Program should compile with `make`, without any parameters. This means the first rule in the Makefile should be the build command.

2\. The program will have to be modified to split the training and inference stages, saving the trained weights to a specified file. This will facilitate the independent execution of each of these stages, crucial given the long execution time of deep learning programs on the simulator. The main function should be modified to accept the following "modes":

#### Train
```console
$ <program> -train <epochs> <weights_file>
```
- `<program>` the program binary
- `-train` specifies that training should be executed
- `<epochs>` number of epochs to train for
- `<weights_file>` the file to save the model's trained weights to


#### Train (Increment)
```console
$ <program> -train-increment <start_epoch> <end_epoch> <input_weights_file> <output_weights_file>
```
- `<program>` the program binary
- `-train-increment` specifies the incremental training mode
- `<start_epoch>` number of epochs that have already been trained
- `<end_epoch>` the epoch number to terminate training
- `<input_weights_file>` the file that contains the weights trained prior to the point of program execution
- `<output_weights_file>` the file to save the weights post this round of training

#### Test
```console
$ <program> -test <weights_file>
```
- `<program>` the program binary
- `-test` specifies that inference should be executed
- `<weights_file>` the file containing the trained weights

Each of the programs in the [`variable-precision-gpu` GitHub organization](https://github.com/variable-precision-gpu) have already been made compatible with the above rules and can serve as examples.

3\. Generate and override PTX accordingly, if using mixed precision

### 3. Runner Execution
Before executing the runner script `runner.py`, configure the execution parameters first. A brief description of what the script does is provided at the top of the file for reference. Specify the target program, the name of the executable and the stages to run.

You can then run the script with:
```console
$ python3 runner.py
```

### 4. Overriding PTX
In order to perform mixed precision deep learning, you first have to run the target program once on the simulator to generate PTX files.

Then, go to the `configs/gpgpusim.config` and set the `-gpgpu_generate_ptx` option to `0` - this turns off PTX generation, and the next time the simulator is run, it will ingest PTX files directly.

Then, edit the PTX files to override PTX instructions, setting the types of the instructions to `VF32`. Instructions marked as `VF32` will be performed at an arbitrary precision, depending on the values of the `VF_SIGNIFICAND`, `VF_EXPONENT_MIN` and `VF_EXPONENT_MAX` environment variables at the time of kernel invocation.

For details on how exactly to perform PTX overriding, including what parameters to set to emulate certain precisions, please refer to Section 4 of the final report.

While you can configure the aforementioned `VF32` environment variables in the `runner.py` script to apply to an entire execution run, if you would like to change `VF32` precision within a run (e.g. between neural network layers), you will have to set the environment variables in the application code itself. For example:
```c++
// set VF32 precision and range to be that of bfloat16
setenv("VF_SIGNIFICAND", "8", 1);
setenv("VF_EXPONENT_MIN", "-132", 1);
setenv("VF_EXPONENT_MAX", "128", 1);
```