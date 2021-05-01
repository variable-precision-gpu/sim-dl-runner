"""GPGPU-Sim Deep Learning Runner Script

This script manages the execution of deep learning programs on GPGPU-Sim.

New deep learning programs should be first made compatible with the simulator
according to the instructions in the README, To use VF32, make sure that the PTX
for the deep learning program has been generated (by running it on the simulator
once) and that you have overriden the PTX to use VF32.

Ensure that the repo configuration parameters accurate reflect your local
project structure. The parameters under script configuration should be set
accordingly to control execution behavior. Briefly, the four stages in this
script are as follows:

    BUILD:
    - Compile DL program
    - Apply the simulator config file
    - Setup simulator environment
    TRAIN:
    - Executes training on the DL program
    - Params:
        - input weights file (optional)
        - output weights file
        - start epoch (optional)
        - end epoch
    INFER:
    - Executes inferences on the DL program
    - Params:
        - weights file
        - log file (optional)
    CLEANUP:
    - Cleans up temporary files

 Each stage can be individually disabled.
"""

from collections import OrderedDict
from time import sleep
import subprocess
import os

# Repo configuration (modify this)
PROGRAM_DIR = "./programs"
CONFIG_DIR = "./config"
SIM_DIR = "./../gpgpu-sim_distribution"

# Script configuration (modify this)
BASE_VF_SIGNIFICAND = 8
BASE_VF_EXPONENT_MIN = -132
BASE_VF_EXPONENT_MAX = 128
PROGRAM = "MLP2"
EXECUTABLE = "mlp2_sim"
STAGE_CONFIG = {
    "BUILD": {
        "RUN": True,
    },
    "TRAIN": {
        "RUN": False,
        "START_EPOCH": 1,
        "INPUT_WEIGHTS_FILE": "",
        "END_EPOCH": 10000,
        "OUTPUT_WEIGHTS_FILE": "weights10000.txt",
    },
    "TEST": {
        "RUN": True,
        "WEIGHTS_FILE": "weights10000.txt",
        "LOG_FILE": "",
    },
    "CLEANUP": {
        "RUN": True,
    },
}

def stream(process):
    """
    Streams a given process' stdout to this process' stdout.
    """

    result = None
    while result is None:
        for line in process.stdout:
            print(line.decode(), end="")
        for line in process.stderr:
            print(line.decode(), end="")
        result = process.poll()
        sleep(0.1)
    return result


def metaprint(stage, message):
    """
    Helper function to print "meta" messages about the script execution.

    Messages will be printed in green.
    """

    string = "[{}] {}".format(stage, message)
    META_COLOR = '\033[92m'
    END_COLOR = '\033[0m'
    print(META_COLOR + string + END_COLOR)


def run(command, env=os.environ):
    """
    Runs a given command in a subprocess.
    """

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.join(PROGRAM_DIR, PROGRAM),
        shell=True,
        executable='/bin/bash',
        env=env)
    result = stream(process)
    if result != 0:
        print("Error: return code {}".format(result))
        exit(result)


def run_with_sim_setup(command):
    """
    Runs a given command in a subprocess, with simulator setup.
    """

    env = os.environ.copy()
    # set path to gpgpusim.config
    env["SIM_CONFIG_PATH"] = os.path.abspath(
        os.path.join(CONFIG_DIR, "gpgpusim.config"))
    # set base precision of VF32 type
    env["VF_SIGNIFICAND"] = str(BASE_VF_SIGNIFICAND)
    env["VF_EXPONENT_MIN"] = str(BASE_VF_EXPONENT_MIN)
    env["VF_EXPONENT_MAX"] = str(BASE_VF_EXPONENT_MAX)
    # source sim setup file
    setup_file = os.path.abspath(os.path.join(SIM_DIR, "setup_environment"))
    command = ". {}; {};".format(setup_file, command)
    run(command, env)


def build():
    """
    Invoke the pre-determined build command for DL applications.
    """

    run("make")


def train():
    """
    Launches training on the simulator for the configured application.

    If a start epoch greater than 1 is provided, a weights input file is expected
    to perform incremental training.
    """

    config = STAGE_CONFIG["TRAIN"]
    if config["START_EPOCH"] <= 1:
        # run regular training
        run_with_sim_setup("./{} -train {} {}".format(EXECUTABLE,
                                                      config["END_EPOCH"],
                                                      config["OUTPUT_WEIGHTS_FILE"]))
    else:
        # run incremental training, starting from an input weights file
        run_with_sim_setup("./{} -train-increment {} {} {} {}".format(EXECUTABLE,
                                                                      config["START_EPOCH"],
                                                                      config["END_EPOCH"],
                                                                      config["INPUT_WEIGHTS_FILE"],
                                                                      config["OUTPUT_WEIGHTS_FILE"]))


def test():
    """
    Launches inference on the simulator for the configured application.

    If a non-empty log file name is provided, the simulator output is piped
    to the log file.
    """

    config = STAGE_CONFIG["TEST"]
    command = "./{} -test {}".format(EXECUTABLE,
                                     config["WEIGHTS_FILE"])
    # pipe simulator output to log file (if any)
    log_file = config["LOG_FILE"]
    if log_file != "":
        command = "{} > {}".format(command, log_file)
    run_with_sim_setup(command)


def cleanup():
    """
    Deletes temporary files created by simulator.
    """

    run("rm -f _cuobjdump_list_ptx_* _app_cuda_version_*")


if __name__ == "__main__":
    STAGE_FUNC = OrderedDict([
        ("BUILD", build),
        ("TRAIN", train),
        ("TEST", test),
        ("CLEANUP", cleanup),
    ])
    assert(STAGE_FUNC.keys() == STAGE_CONFIG.keys())

    for stage, func in STAGE_FUNC.items():
        if STAGE_CONFIG[stage]["RUN"]:
            metaprint(stage, "Starting...")
            func()
            metaprint(stage, "Completed!")
        else:
            metaprint(stage, "Skipped")
