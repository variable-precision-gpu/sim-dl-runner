from collections import OrderedDict
from time import sleep
import subprocess
import os

# Repo configuration (modify this)
PROGRAM_DIR = "./programs"
CONFIG_PATH = "./../../config/gpgpusim.config"  # TODO: Fix relative paths
SIM_SETUP_FILE = "./../gpgpu-sim_distribution/setup_environment"

# Script configuration (modify this)
BASE_VF32_SIGNIFICAND = 24
PROGRAM = "MLP"
EXECUTABLE = "mlp_gpgpu"
STAGE_CONFIG = {
    "SETUP_SIM": {
        "RUN": True,
    },
    "BUILD": {
        "RUN": True,
    },
    "TRAIN": {
        "RUN": True,
        "EPOCHS": 1,
        "WEIGHTS_FILE": "weights.csv",
    },
    "TEST": {
        "RUN": False,
        "WEIGHTS_FILE": "weights.csv",
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


def setup_sim():
    # result = subprocess.run(SIM_SETUP_FILE, stdout=subprocess.PIPE,
    #                         stderr=subprocess.PIPE, shell=True, check=True)
    # print(result.stdout.decode(), end="")
    # print(result.stderr.decode(), end="")
    os.system(". " + SIM_SETUP_FILE)
    os.environ["SIM_CONFIG_PATH"] = CONFIG_PATH
    os.environ["VF_SIGNIFICAND"] = str(BASE_VF32_SIGNIFICAND)


def build():
    result = subprocess.run(
        ["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.join(PROGRAM_DIR, PROGRAM), check=True)
    print(result.stdout.decode(), end="")
    print(result.stderr.decode(), end="")


def train():
    process = subprocess.Popen(
        ["./" + EXECUTABLE, "-both", "mnist_train.csv", "mnist_test.csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.join(PROGRAM_DIR, PROGRAM))

    result = stream(process)
    if result != 0:
        exit(result)


def test():
    pass


if __name__ == "__main__":
    STAGE_FUNC = OrderedDict([
        ("SETUP_SIM", setup_sim),
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
