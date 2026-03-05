
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



class CACTUS_VAE_tabular(BaseModel):
    
    def __init__(self,data, params:dict,**kwargs):
        
        #General params
        self.name        = params.get("name","CACTUS_VAE_tabular")
        self.params      = params
        #Data params
        self.data = data
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.nFeatures = data.X_train.shape[-1]
        self.class_out_att = params.get("class_out")
        self.alpha             = params.get('alpha',0.5)
        self.gamma             = params.get('gamma',2)
        self.initial_epoch_C   = params.get('initial_epoch_C',0) 
        self.final_epoch_C     = params.get('final_epoch_C',10)  
        self.slope_C           = params.get('slope_C',1)  

        self.y_class_train = data.context_train[self.class_out_att].values

        #Net params
        self.lt_dim = params.get("lt_dim",32)
        self.units = params.get("units",[32,32])

        #Training params 
        self.epochs = params.get('epochs',2)
        self.patience = params.get('patience',50)
        self.batch_size = params.get('batch_size',64)
        self.training_hist =None
        self.model =None      
        self.capacity_callback = capacity_cb(self.slope_C,0,self.initial_epoch_C,self.final_epoch_C)


        # Initializers
        self.initializer_relu = initialiazers.VarianceScaling()
        self.initializer_linear = initialiazers.RandomNormal(0.,0.02)
        
        # Model
        print(f"Building model: {CACTUS_VAE_tabular.get_model_name()} [{CACTUS_VAE_tabular.get_model_type()}]")
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
        self.enc_out = self._createEnc(self.main_input_layer)

        #Bottleneck
        self.z_m = layers.Dense(units=self.lt_dim,
                            activation = None,
                            kernel_initializer= self.initializer_linear ,
                            bias_initializer= self.initializer_linear,
                            name="z_m" )(self.enc_out)
        
        self.z_log = layers.Dense(units=self.lt_dim,
                            activation = None,
                            kernel_initializer= self.initializer_linear ,
                            bias_initializer= self.initializer_linear,
                            name="z_log" )(self.enc_out)
        
        #dims = self.z_m.shape[1:]
        self.z = self.sampling(self.z_m, self.z_log)

        self.z_input = layers.Input(dtype = tf.float32,shape=self.z.shape.as_list()[1:],name='z_input')
        
        #### CLASSIFICATION #####
        #self.z_in_class = layers.Flatten()(self.z_input)
        self.hidden_class =self.z_input
        #self.hidden_class = layers.Dense(units = 10,activation='tanh',name="hidden_class")(self.hidden_class) 
        self.class_out = layers.Dense(units = self.y_class_train.shape[-1],activation='sigmoid',name="out_class")(self.hidden_class)
        
        # Decoder
        self.y = self._createDec(self.z_input)

        #Output
        self.y = layers.Dense(units= self.nFeatures,
                                activation= None,
                                kernel_initializer= self.initializer_linear ,
                                bias_initializer= self.initializer_linear,
                                name="output" )(self.y)
        
        # Keras models
        self.model_enc = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.z_m])
        self.model_dec = tf.keras.Model(inputs=[self.z_input],outputs = [self.y])
        self.classifier = tf.keras.Model(inputs=[self.z_input],outputs = [self.class_out])
        

        self.model = tf.keras.Model(inputs=[self.main_input_layer],outputs = [self.model_dec(self.model_enc(self.main_input_layer)),self.classifier(self.model_enc(self.main_input_layer))])

        self.model_training =  BetaVAEModelTrainStep(inputs= self.main_input_layer,
                                           outputs=[self.model_dec(self.model_enc(self.main_input_layer)),
                                                    self.z_m,
                                                    self.z_log,
                                                    self.classifier(self.model_enc(self.main_input_layer))],
                                           gamma = self.gamma,
                                           alpha = self.alpha 
                                           )

        self.model_training.compile( optimizer='adam',run_eagerly=True)
        
        if verbose:
            self.model.summary()        
        
        return None

    def train(self):
        print(f"Training model: {CACTUS_VAE_tabular.get_model_name()} [{CACTUS_VAE_tabular.get_model_type()}]")
        
        train_X, v_X, train_class,v_class =  train_test_split(self.X_train, self.y_class_train, test_size=.15,random_state=10)  
        ES_cb = tf.keras.callbacks.EarlyStopping( monitor="val_loss",
                                        min_delta=0.001,
                                        patience=self.patience,                                            
                                        baseline=None,
                                        restore_best_weights=True)
        
        self.training_hist=self.model_training.fit([train_X], [train_X,train_class] ,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=([v_X], [v_X,v_class]),
                        callbacks=[self.capacity_callback])
    
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

        print(f"saving weights of model {CACTUS_VAE_tabular.get_model_name()}: {self.name}")
        self.model.save_weights(savePath+f"/{self.name}")


        self.training_data.to_csv(os.path.join(savePath,"training_data.csv"))
        return None

    def load(self):
        
        if self.model==None:
            raise Exception("The model has not been defined yet")
    
        loadPath = os.path.join(path_weights, f"{self.name}")
        print(f"restoring weights of model {CACTUS_VAE_tabular.get_model_name()}: {self.name}")
        self.model.load_weights(loadPath+f"/{self.name}")
        self.training_data = pd.read_csv(os.path.join(loadPath,"training_data.csv"))
        return None



    def sampling(self,z_m,z_log):        
        batch = tf.shape(z_m)[0]
        dim = tf.shape(z_m)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_m + tf.exp(0.5 * z_log) * epsilon
    
    @classmethod
    def get_model_type(cls):
        return "gen" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CACTUS_VAE_tabular" # Aquí se puede indicar un ID que identifique el modelo
    
    @classmethod
    def target(cls):
        return "single-target" 
    @classmethod
    def tast(cls):
        return "single-task" 

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 


class BetaVAEModelTrainStep(tf.keras.Model):
    def __init__(self,alpha,gamma,**kwargs):
        super(BetaVAEModelTrainStep, self).__init__(**kwargs) 
        # Tensor for the capacity
        self.Kappa = tf.Variable(0,dtype='float32',trainable=False)
        self.gamma = gamma
        self.alpha = alpha
        
    
    def train_step(self, data):
        _,(x,y)=data
        sample_weight = None
        with tf.GradientTape() as tape:
            tape.watch(self.Kappa)
            x_pred,z_m,z_log,y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            kl_loss = -0.5 * (1 + z_log - K.square(z_m) - K.exp(z_log))
            kl_loss = K.sum(kl_loss, axis=1, keepdims=True)
            kl_loss = K.mean(kl_loss)
            kl_loss =self.gamma* K.abs(kl_loss - self.Kappa)

            reco_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x,x_pred))
            
            cls_loss = tf.keras.losses.BinaryCrossentropy()(y,y_pred)

            loss = self.alpha*(reco_loss + kl_loss) + (1-self.alpha)*cls_loss
            

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"loss": loss, "reco_loss": reco_loss,"kl_loss": kl_loss,"cls_loss":cls_loss, "beta":self.Kappa}

    def test_step(self, validation_data):        
        _,(x,y) = validation_data
        x_pred,z_m,z_log,y_pred = self(x, training=False)
        kl_loss_val = -0.5 * (1 + z_log - tf.square(z_m) - tf.exp(z_log))
        kl_loss_val = tf.reduce_mean(tf.reduce_sum(kl_loss_val,axis=-1)) 
        reco_loss_val = tf.reduce_mean(tf.keras.losses.mean_squared_error(x,x_pred))
        cls_loss_val = tf.keras.losses.BinaryCrossentropy()(y,y_pred)
        loss_val = self.alpha*(reco_loss_val + kl_loss_val) + (1-self.alpha)*cls_loss_val

        return {"loss": loss_val} # just the reconstruction



# Class to control the capacity of beta-VAE  during training
class capacity_cb(tf.keras.callbacks.Callback):
    def __init__(self, slope:float,capacity_0:float, start_epoch_capacity:int,end_epoch_capacity:int):
        """
        PARAMS:
            slope (float):              rate of capacity increase
            capacity_0 (float):         capacity for epoch 0
            start_epoch_capacity (int): epoch from which the capacity will be increased
            end_epoch_capacity (int):   epoch from which the the increment of the capacity will stop

        OUTPUT
            Keras callback that gradually increase the value of capacity stored in a TF variable. 
        """
        super(capacity_cb).__init__()
        self.capacity_0=capacity_0
        self.slope = slope
        self.val_capacity = capacity_0
        self.start_epoch = start_epoch_capacity
        self.end_epoch = end_epoch_capacity
        
        
    def on_epoch_end(self,epoch, logs=None):
        if (epoch >= self.start_epoch) and (epoch <= self.end_epoch):
            self.val_capacity = max(0,epoch*self.slope + self.capacity_0)
            
        # The variable must be defined in the model. 
        K.set_value(self.model.Kappa, tf.constant(self.val_capacity))
        #TODO: This callback must be coordinated with the model in order to avoid future inconsistent errors beacuse
        #      of the capacity variable's name