from collections import OrderedDict
from time import sleep
import subprocess
import os

# Repo configuration (modify this)
PROGRAM_DIR = "./programs"
CONFIG_DIR = "./config"
SIM_DIR = "./../gpgpu-sim_distribution"

# Script configuration (modify this)
BASE_VF32_SIGNIFICAND = 24
PROGRAM = "MLP"
EXECUTABLE = "mlp_cuda"
STAGE_CONFIG = {
    "BUILD": {
        "RUN": True,
    },
    "TRAIN": {
        "RUN": False,
        "START_EPOCH": 250,
        "INPUT_WEIGHTS_FILE": "weights2.csv",
        "END_EPOCH": 500,
        "OUTPUT_WEIGHTS_FILE": "weights3.csv",
    },
    "TEST": {
        "RUN": True,
        "WEIGHTS_FILE": "weights4.csv",
        "LOG_FILE": "",
    },
}

# Program constants (do not modify)
META_COLOR = '\033[92m'
END_COLOR = '\033[0m'


def stream(process):
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
    string = "[{}] {}".format(stage, message)
    print(META_COLOR + string + END_COLOR)


def run(command, env=os.environ):
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
    env = os.environ.copy()
    # set path to gpgpusim.config
    env["SIM_CONFIG_PATH"] = os.path.abspath(
        os.path.join(CONFIG_DIR, "gpgpusim.config"))
    # set base precision of VF32 type
    env["VF_SIGNIFICAND"] = str(BASE_VF32_SIGNIFICAND)
    # source sim setup file
    setup_file = os.path.abspath(os.path.join(SIM_DIR, "setup_environment"))
    command = ". {}; {};".format(setup_file, command)
    run(command, env)


def build():
    run("make")


# TODO: Implement incremental training
def train():
    config = STAGE_CONFIG["TRAIN"]
    if config["START_EPOCH"] <= 1:
        run_with_sim_setup("./{} -train {} {}".format(EXECUTABLE,
                                                      config["END_EPOCH"],
                                                      config["OUTPUT_WEIGHTS_FILE"]))
    else:
        run_with_sim_setup("./{} -train-increment {} {} {} {}".format(EXECUTABLE,
                                                                      config["START_EPOCH"],
                                                                      config["END_EPOCH"],
                                                                      config["INPUT_WEIGHTS_FILE"],
                                                                      config["OUTPUT_WEIGHTS_FILE"]))


def test():
    run_with_sim_setup("./{} -test {}".format(EXECUTABLE,
                                              STAGE_CONFIG["TEST"]["WEIGHTS_FILE"]))


if __name__ == "__main__":
    STAGE_FUNC = OrderedDict([
        ("BUILD", build),
        ("TRAIN", train),
        ("TEST", test)
    ])
    assert(STAGE_FUNC.keys() == STAGE_CONFIG.keys())

    for stage, func in STAGE_FUNC.items():
        if STAGE_CONFIG[stage]["RUN"]:
            metaprint(stage, "Starting...")
            func()
            metaprint(stage, "Completed!")
        else:
            metaprint(stage, "Skipped")
