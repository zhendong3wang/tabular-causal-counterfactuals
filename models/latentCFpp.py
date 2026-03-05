
import ipdb
import tensorflow as tf
from address import *
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from .CNN import CNN
from .BaseModel import BaseModel
import numpy as np



class latentCFpp():
    """
        Counterfactual generator LatentCF:

           Learning Time Series Counterfactuals via Latent Space Representations,Wang, Z., Samsten, I., Mochaourab, R., Papapetrou, P., 2021.
            in: International Conference on Discovery Science, pp. 369–384. https://doi.org/10.1007/978-3-030-88942-5_29

            - https://github.com/zhendong3wang/learning-time-series-counterfactuals/blob/971a3dad294ef1322b713e5dabd66a5eff327441/src/_composite.py

    """
    def __init__(self, classifier:CNN, gen:BaseModel, params: dict, **kwargs)->None:

        # CF params and models 
        
        self.epochs            = params.get('epochs',30)
        self.tol               = params.get('tol',0.01)
        self.target_prob       = params.get('target_prob',0.9)
        self.loss              = params.get("loss",self._loss)
        self.learning_rate     = params.get("learning_rate",1e-2)
        
        self.loss_history      = []
        
        self.classifierModel   =  classifier
        self.genModel          =  gen
        self.nb_cls            = self.classifierModel.nb_clss

        self.classifier        = self.classifierModel.model
        self.decoder           = self.genModel.model_dec
        self.encoder           = self.genModel.model_enc
        self.z_input           = self.genModel.z_input

        self._mse_loss         = MeanSquaredError()
        self.target_prob       = tf.constant([self.target_prob]) 
        
        self.optimizer         = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Building the CF generation model
        self.fit()



    def fit(self):
        """
            Creates the TF graph and the loss function to generate the CF sample
        """
        print(f"Building model: {latentCFpp.get_model_name()} [{latentCFpp.get_model_type()}]")
        #ipdb.set_trace()
        self.model_z_to_y = Model(inputs=[self.z_input],outputs = [self.classifier(self.decoder(self.z_input))])
        self.model_z_to_y.summary()


        return None       
    
    #TODO: It only transforms one sample at a time. 
    #      It contemplates the binary case
    def transform(self,X,pred_labels,verbose=1,**kwargs):
        """
            Generates the CF. The initial z is iteratively transformed to minimize the target loss. 
            PARAMETERS
                X [numpy array]             -> Input sample to be transformed
                pred_labels [numpy array]   -> Estimated labels for X
            RETURN
                CF [numpy array] -> Estiamted counterfactuals
        """
        N_samples = X.shape[0]
        CFs=[]
        CFs_y_ = []
        for i_sample in range(N_samples):
            print(f"Generating CF {i_sample}/{N_samples}")
            z = tf.Variable(self.encoder(X[[i_sample]]))
            y_logit = self.classifier(X[[i_sample]])
            
            target_cls = 1-pred_labels[i_sample]
            epoch = 0 
            y_logit = y_logit.numpy().squeeze()
            y_ = np.argmax(y_logit)
            while (y_logit[target_cls] < self.target_prob) and (self.epochs > epoch):

                with tf.GradientTape() as tape:
                    tape.watch(z)
                    loss = self._loss(z,target_cls)

                # Getting and applying the gradient wrt z
                grad = tape.gradient(loss,z)
                self.optimizer.apply_gradients([(grad,z)])
                # Computing prediction and decoding z
                y_logit = self.model_z_to_y(z)
                
                y_logit = y_logit.numpy().squeeze()
                y_ = np.argmax(y_logit)
                cf_sample = self.decoder(z)
                epoch = epoch+1 

                if verbose>0:
                    print(f"epoch {epoch}/{self.epochs} Target {self.target_prob} Target cls {target_cls} \t Loss tol. {self.tol}") 
                    print(f"Pred logit {y_logit}   Pred {y_}  Loss {loss}")
                    print(f" Grad. abs/mean: {np.mean(np.abs(grad)):.2f}/{np.mean(grad):.2f}  Grad. min/max: {np.min(grad):.2f}/{np.max(grad):.2f} \n") 

            CFs.append(cf_sample.numpy())
            CFs_y_.append(y_)
        
        return np.concatenate(CFs,axis=0),np.array(CFs_y_),X,pred_labels
    

    def test_z_y(self,z):
        return self.model_z_to_y.predict(z)

    @tf.function
    def _loss(self,z_k, target_class):
        """
        Loss function for CF generation

        PARAMS
            original_sample [tf.variable] --> The original sample before start searching
            z_k [tf.Variable]             --> The z point 
            target_clas [int]             --> Target class of the final counterfactual  

        """
        
        y_ = self.model_z_to_y(z_k)[:,target_class]
        return tf.reduce_mean(tf.square(self.target_prob-y_)) 

    @classmethod
    def get_model_type(cls):
        return "CF" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "latentCF" # Aquí se puede indicar un ID que identifique el modelo

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 