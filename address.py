import os
import json

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Data/'
results_grid_search = str(path_here)+'/Results/Params/'
path_results_metrics = str(path_here)+'/Results/'
path_models = str(path_here)+'/Models/'
path_weights = str(path_here)+'/exp/'

def get_param_path(modelID):
    return os.path.join(results_grid_search,modelID+'.csv')

def get_data_path(dataID):
    return os.path.join(dataset_dir,dataID+'/')


def get_model_path(modelID):
    return os.path.join(path_models,modelID+'/')

def get_weights_path(modelID):
    return os.path.join(path_weights,modelID+'/')

def get_exp(name):
    path = os.path.join(path_weights,name) +'/params.json'
    with open(path) as params_file:
        params = json.load(params_file)
    return params
def get_exp_config(path):
    with open(path) as params_file:
        params = json.load(params_file)
    return params
