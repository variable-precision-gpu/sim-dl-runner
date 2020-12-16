# GPGPU-Sim DL Runner

This program manages the execution of deep learning programs on GPGPU-Sim.

## Requirements
- [x] Compile programs
- [x] Apply a global sim config file to all programs
- [x] Setup sim environment
- [ ] Train  
Params: weights, activations, biases, epochs, start epoch, end epoch
- [ ] Infer  
Params: weights, logs
- [ ] Clean up intermediate files (_app_cuda_version, _cuobjdump_list_ptx)

## Usage
### Making DL Program Compatible
- Program needs to be compilable with `make`
- Compiled binary needs to have the same name as the program
### Configuring and Executing Script
Configure:
- program directory
- sim setup path
- config path
