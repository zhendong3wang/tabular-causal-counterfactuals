
from .DataReader_GivmeCred import DataReader_GivmeCred
from .DataReader_Adult import DataReader_Adult
from .DataReader_Law import DataReader_Law
from .DataReader_GermanCredit import DataReader_GermanCredit
from .DataReader_HELOC import DataReader_HELOC
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from address import *
from utils import upsampling

def getData(params):
    
    
    if params["data"] == "GIVECREDIT":
        PATH = get_data_path("GiveMeCredit")
        data = DataReader_GivmeCred(PATH)

        data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)

        data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    elif params["data"] == "ADULT":
        PATH = get_data_path("Adult")
        data = DataReader_Adult(PATH)

        # data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)
        # data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    elif params["data"] == "LAW":
        PATH = get_data_path("Law")
        data = DataReader_Law(PATH)

        # data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)
        # data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    elif params["data"] == "GERMANCREDIT":
        PATH = get_data_path("GermanCredit")
        data = DataReader_GermanCredit(PATH)

        # data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)
        # data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    elif params["data"] == "HELOC":
        PATH = get_data_path("HELOC")
        data = DataReader_HELOC(PATH)

        # data.X_train, data.y_train, up_idx = upsampling(data.X_train, data.y_train)
        # data.context_train = data.context_train.iloc[up_idx,:]
        
        return data
    else:
        raise Exception(f"Dataset is not supported yet")