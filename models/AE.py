
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
from address import *
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
import json



class AE(BaseModel):
    
    def __init__(self,data, params:dict,**kwargs):
        
        #General params
        self.name        = params.get("name","AE")
        self.params      = params
        #Data params
        self.data = data
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.nFeatures = data.X_train.shape[-1]

        #Net params
        self.lt_dim = params.get("lt_dim",32)
        self.units = params.get("units",[32,32])

        #Training params 
        self.epochs = params.get('epochs',2)
        self.patience = params.get('patience',20)
        self.batch_size = params.get('batch_size',64)
        self.training_hist =None
        self.model =None      

        # Initializers
        self.initializer_relu = initialiazers.VarianceScaling()
        self.initializer_linear = initialiazers.RandomNormal(0.,0.02)
        
        # Model
        print(f"Building model: {AE.get_model_name()} [{AE.get_model_type()}]")
        self.create_model()



    def _createEnc(self,x):
        self.layerEnc = []
        
        for ll,u in enumerate(self.units):
            self.layerEnc.append(layers.Dense(units=u,
                                              activation='relu',
                                              kernel_initializer=self.initializer_relu,
                                              bias_initializer=self.initializer_relu,
                                              name = f"Enc_layer{ll}"))
        
        #Building encoder
        y = x
        for layer in self.layerEnc:
            y = layer(y)
        return y

    def _createDec(self,z):
        self.layersDec = []
        for ll,u in enumerate(reversed(self.units)):
            self.layersDec.append(layers.Dense(units=u,
                                              activation='relu',
                                              kernel_initializer=self.initializer_relu,
                                              bias_initializer=self.initializer_relu,
                                              name = f"Dec_layer{ll}"))
        #Building decoder
        y = z
        for layer in self.layersDec:
            y = layer(y)
        return y


    def create_model(self,verbose=True):
        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.nFeatures,],name='main_input')
        # AE 
        # Encoder 
        self.z = self._createEnc(self.main_input_layer)

        #Bottleneck
        self.z = layers.Dense(units=self.lt_dim,
                            activation = None,
                            kernel_initializer= self.initializer_linear ,
                            bias_initializer= self.initializer_linear,
                            name="z" )(self.z)
        

        self.z_input = layers.Input(dtype = tf.float32,shape=self.z.shape.as_list()[1:],name='z_input')
        
        
        # Decoder
        self.y = self._createDec(self.z_input)

        #Output
        self.y = layers.Dense(units= self.nFeatures,
                                activation= None,
                                kernel_initializer= self.initializer_linear ,
                                bias_initializer= self.initializer_linear,
                                name="output" )(self.y)
        
        # Keras models
        self.model_enc = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.z])
        self.model_dec = tf.keras.Model(inputs=[self.z_input],outputs = [self.y])

        

        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.model_dec(self.model_enc(self.main_input_layer))])



        self.model.compile( optimizer='adam',
                      loss=[tf.keras.losses.MeanSquaredError()],
                      loss_weights=[1]
                    )
        
        if verbose:
            self.model.summary()        
        
        return None

    def train(self):
        print(f"Training model: {AE.get_model_name()} [{AE.get_model_type()}]")
        
        train_X, v_X=  train_test_split(self.X_train, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model.fit([train_X], [train_X] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=([v_X], [v_X]),
                        callbacks=[ES_cb])
    
        self.training_data = pd.DataFrame(self.training_hist.history)
        return self.training_hist 





    def predict(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                X_ [numpy array] -> Estiamted input sample
        """

        o,_ = self.model.predict(X)
        return o
  

    def predict_enc(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                z[numpy array] -> Bottleneck
        """


        return self.model_enc.predict(X)

    def store(self):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        
        savePath = os.path.join(path_weights, f"{self.name}")
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        print(f"saving weights of model {AE.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")


        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None

    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {AE.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None

    @classmethod
    def get_model_type(cls):
        return "gen" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "AE" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def target(cls):
        return "single-target" 
    @classmethod
    def tast(cls):
        return "single-task" 

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 


