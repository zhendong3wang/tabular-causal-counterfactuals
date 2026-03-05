# SCRIPT FOR TRAINING ALL THE MODELS
# It takes as input variable the path the config file (json).

import ipdb
from Data import *
from address import *
from modelGen import *
from utils import upsampling
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = get_exp_config(args.config)
    print(config)
    data = getData(config)


    model = modelGen(config["type"],data,params=config,debug=True)
    model.train()
    model.store()
    #model.load()



