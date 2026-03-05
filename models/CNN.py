from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from .BaseModel import BaseModel
from address import *
import ipdb
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC,CategoricalAccuracy
import pandas as pd
import json

class CNN(BaseModel):

    def __init__(self, data, params: dict, **kwargs)->None:
        # Data params 
        self.leads = data.lead_labels
        
        

        # Data
        self.params = params
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.training_hist =None
        self.model =None

        # Net params
        self.sequence_length    = params.get('sequence_length',self.X_train.shape[1])
        self.nChannels          = params.get('nChannels',self.X_train.shape[-1])
        self.cls_labels         = params.get('clss',["STTC", "NORM", "MI", "HYP", "CD"])
        self.nb_clss            = len(self.cls_labels)
        self.cls_labels_all     = data.cls_lbls_all
        self.name               = params.get("name","CNN")

        # No net params. ALl the hyperparams are set according to
        #   https://github.com/Paulet306/Counterfactual-Explanations-for-12-lead-ECG-classification/tree/main 

        # Training params
        self.epochs     = params.get('epochs', 2)
        self.patience   = params.get('patience', 15)
        self.batch_size = params.get('batch_size',64)        

        # Metrics
        self.metrics    = [CategoricalAccuracy(), AUC()]

        # Model
        print(f"Building model: {CNN.get_model_name()} [{CNN.get_model_type()}]")
        self.create_model()

    def create_model(self,verbose=True):
        # Input layers
        self.main_input_layer = layers.Input(dtype = tf.float32,shape=[self.sequence_length,self.nChannels],name='main_input')
        
        # Defining the model
        self.layers = []
        self.layers.append(layers.Conv1D(filters=64,
                                        kernel_size=8,
                                        padding='same',
                                        activation='relu',
                                        kernel_regularizer=l2(0.001)))
        self.layers.append(layers.BatchNormalization())
        self.layers.append(layers.Dropout(0.1))

        self.layers.append(layers.Conv1D(filters=128,
                                  kernel_size=5,
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(0.001)))
        self.layers.append(layers.BatchNormalization())
        self.layers.append(layers.Dropout(0.2))        

        self.layers.append(layers.Conv1D(filters=256,
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu',
                                  kernel_regularizer=l2(0.001)))
        self.layers.append(layers.BatchNormalization())
        self.layers.append(layers.Dropout(0.3))  

        self.layers.append(layers.GlobalAveragePooling1D())
        self.layers.append(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        self.layers.append(layers.Dropout(0.3)) 

        
        self.layers.append(layers.Dense(self.nb_clss,activation='softmax'))
        

        # Building the model
        self.y = self.main_input_layer
        for layer in self.layers:
            self.y = layer(self.y)    
        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.y])
        # Compiling the model 
        self.model.compile( optimizer='adam',
                        loss="categorical_crossentropy",
                        metrics=self.metrics)

        if verbose:
            self.model.summary()

        return None       
    
    
    def train(self):
        print(f"Training model: {CNN.get_model_name()} [{CNN.get_model_type()}]")
        train_X, v_X, train_y, v_y =  train_test_split(self.X_train, self.y_train, test_size=.15,random_state=10)  
        
        
        
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model.fit([train_X], [train_y] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(v_X, v_y),
                        callbacks=[ES_cb])
        
        self.training_data = pd.DataFrame(self.training_hist.history)
    def predict(self,X):
        """
            It predicts a unique sample
            PARAMETERS
                X [numpy array]  -> Input sample 
            RETURN
                y_ [numpy array] -> Estiamted class labels
        """
        return self.model.predict(X) 
    
    def store(self):
        if self.training_hist==None:
            raise Exception("The model has not been trained yet")
        
        savePath = os.path.join(path_weights, f"{self.name}")
        if not os.path.exists(savePath):
            os.mkdir(savePath)

        print(f"saving weights of model {CNN.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")

        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None
    
    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {CNN.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None
    

    @classmethod
    def get_model_type(cls):
        return "class" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CNN" # Aquí se puede indicar un ID que identifique el modelo

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 