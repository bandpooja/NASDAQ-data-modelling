import os
from pathlib import Path

from utils.helper import get_constants

if __name__ == "__main__":
    constants = get_constants()

    # make all the desired directory in the docker environment
    for k in constants.keys():
        Path(constants[k]).mkdir(parents=True, exist_ok=True)

    # download the pickle files needed to load the models
    # uploaded to  ref:

    os.system("")
