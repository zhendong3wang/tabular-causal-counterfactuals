from models import *
import pandas as pd
from address import *
import json


# Model factory pattern
def modelGen(modelID:str,data,params:dict={},verbose=True,debug = False):
    '''
    ARGUMENTS
        modelID (str)                       ID that indicates the model type
        data    (featExtraction object)     Data object needed to train
        params  (dict)                      the params that define the model 
    '''
    data = data
    modelID = modelID
    params  = params

    if verbose:
        print("Building model")

    if not params and not debug:
        if verbose:
            print("loading best hyperparameters")
        params_path  = get_param_path(modelID)
        
        with open(params_path) as f:
            params = json.load(f)
        
        #df_params    = pd.read_csv(params_path,index_col=0)
        #params       = ast.literal_eval(df_params.loc[data.dataID,'params'])[0]


    #TODO: Make it more generic: https://stackoverflow.com/questions/456672/class-factory-in-python 
    for cls in BaseModel.__subclasses__():
        #print(cls.get_model_name())
        if cls.is_model_for(modelID):
            return cls(data,params)
    raise Exception("Model not implemented")
    
    


if __name__ == "__main__":
    import ipdb
    from Data import *
    from address import *

    PATH = get_data_path("PTB-XL")
    data = DataReader(PATH,100)
    print("reading data")
    model = modelGen("FCNAE",data,{},debug=True)
    model.train()
    


    

    
