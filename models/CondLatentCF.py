
import ipdb
import tensorflow as tf
from address import *
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanAbsoluteError,BinaryCrossentropy,CategoricalCrossentropy, MeanSquaredError
from .CNN import CNN
from .BaseModel import BaseModel
import numpy as np
from itertools import product
from utils import polynomial_decay


class CondLatentCF():
    """
        Counterfactual generator using a conditioned Latent Space

    """
    def __init__(self, classifier:CNN, gen:BaseModel, params: dict, **kwargs)->None:

        # CF params and models 
        #self.optimizer         = params.get('optimizer',tf.optimizers.Adam(learning_rate=0.2))
        
        self.epochs            = params.get('epochs',30)
        self.tol               = params.get('tol',0.01)
        self.target_prob       = params.get('target_prob',0.9)
        self.alpha             = params.get('alpha',0.5)
        self.beta              = params.get('beta',0.1)
        self.learning_rate     = params.get('learning_rate',0.01)
        self.power             = params.get('power',2)
        self.gamma             = params.get('gamma',1)
        self.loss              = params.get("loss",self._loss)
        self.dynamicAlpha      = params.get("dynamicAlpha",False)

        self.loss_history      = []


        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                            initial_learning_rate=self.learning_rate,
                            decay_steps=self.epochs,
                            end_learning_rate=0.00,
                            power=self.power
                        )
        self.optimizer         = tf.optimizers.Adam(learning_rate=self.learning_rate)
        # if  self.dynamicAlpha:
        #     self.optimizer         = tf.optimizers.Adam(learning_rate=self.learning_rate)
        # else:
        #     self.optimizer         = tf.optimizers.Adam(learning_rate=self.lr_schedule)
        
        self.classifierModel   =  classifier
        self.genModel          =  gen
        self.nb_cls            = self.classifierModel.nb_clss

        self.classifier        = self.classifierModel.model
        self.decoder           = self.genModel.model_dec
        self.encoder           = self.genModel.model_enc
        self.context_classifier       = self.genModel.classifier 
        self.z_input           = self.genModel.z_input

        #self._mse_loss         = MeanSquaredError()
        self._binary_crossentropy         = BinaryCrossentropy()
        self._context_loss      = BinaryCrossentropy()
        self._categorical_crossentropy = CategoricalCrossentropy()
        self.target_prob       = tf.constant([self.target_prob]) 
        # Building the CF generation model
        self.fit()



    def fit(self):
        """
            Creates the TF graph and the loss function to generate the CF sample
        """
        print(f"Building model: {CondLatentCF.get_model_name()} [{CondLatentCF.get_model_type()}]")
        #ipdb.set_trace()

        # Freeze pretrained parts
        self.decoder.trainable = False
        self.classifier.trainable = False
        for layer in self.decoder.layers:
            layer.trainable = False 
        for layer in self.classifier.layers:
            layer.trainable = False

        self.model_z_to_y = Model(inputs=[self.z_input],outputs = [self.classifier(self.decoder(self.z_input))])
        self.model_z_to_y.summary()

        return None       
    
    def transform_all_context(self,X,pred_labels,verbose=1):
        """
            Generates the CF for all posible target contexts. The initial z is iteratively transformed to minimize the target loss. 
            PARAMETERS
                X [numpy array]                 -> Input sample to be transformed
                pred_labels [numpy array]       -> Estimated labels for X
                target_context [2D numpy array] -> Labels of the target context 
            RETURN
                CF [numpy array]                -> Estimated counterfactuals
                CFs_y_ [numpy array]            -> Estimated labels for the counterfactual samples
        """        
        CF_x  = []
        CF_y_ = []
        Y_    = []
        X_    = []
        for target_context in product([0.,1.],[0.,1.]):
            cf_x,cf_y_ = self.transform(X,pred_labels,list(target_context))
            CF_x.append(cf_x)
            CF_y_.append(cf_y_)
            Y_.append(pred_labels.reshape((-1,1)))
            X_.append(X)
        return np.vstack(CF_x),np.vstack(CF_y_),np.vstack(X_),np.vstack(Y_)

    #TODO: It only changes one context label
    #      It contemplates the binary case
    def transform(self,X,pred_labels,target_context,verbose=1):
        """
            Generates the CF. The initial z is iteratively transformed to minimize the target loss. 
            PARAMETERS
                X [numpy array]                 -> Input sample to be transformed
                pred_labels [numpy array]       -> Estimated labels for X
                target_context [2D numpy array] -> Labels of the target context 
            RETURN
                CF [numpy array]                -> Estimated counterfactuals
                CFs_y_ [numpy array]            -> Estimated labels for the counterfactual samples
        """
        N_samples = X.shape[0]
        CFs=[]
        CFs_y_ = []
        context = target_context

        
        for i_sample in range(N_samples):
            target_context = tf.convert_to_tensor(context[[i_sample]],dtype="float32")
            z = tf.Variable(self.encoder(X[[i_sample]]))
            

            y_logit = self.classifier(X[[i_sample]]).numpy().squeeze()
            y_pred_label = np.argmax(y_logit)
            target_cls = 1-pred_labels[i_sample]

            cf_ctx_probas = self.context_classifier(z)
            context_loss = self._context_loss(target_context, cf_ctx_probas)
            
            cf_ctx_probas = cf_ctx_probas.numpy().squeeze()
            
            x_0 = tf.convert_to_tensor(X[[i_sample]],dtype="float32")
            best_cf = x_0

            best_proba = 0.3
            best_y_pred_label = y_pred_label
            found_best = False
            best_L_dist = 1e10
            best_context = cf_ctx_probas
            best_L_context = 1e10
            epoch = 0 
            while self.epochs > epoch:
                
                # if self.dynamicAlpha:
                #     # Polinomial decay in alpha
                #     self.alpha = polynomial_decay(initial_lr=1,
                #                                 end_lr=0,
                #                                 max_epochs=self.epochs,
                #                                 power = 3,
                #                                 current_epoch=epoch)


                with tf.GradientTape() as tape:
                    tape.watch(z)
                    loss = self._loss(x_0,z,target_cls,target_context)

                # Getting and applying the gradient wrt z
                grad = tape.gradient(loss,z)
                z = z - self.learning_rate*grad/tf.reduce_max(tf.abs(grad))
                

                cf_sample = self.decoder(z)

                # FISTA proximal (L1 opt.) 
                delta = cf_sample -x_0
                delta = tf.where(tf.abs(delta) < self.beta, 0, delta)
                cf_sample = x_0 + delta
                
                # Computing prediction and decoding z
                #z = tf.Variable(self.encoder(cf_sample))
                y_logit_ = self.model_z_to_y(z)
                y_logit  = y_logit_.numpy().squeeze()
                y_pred_label = np.argmax(y_logit)
                proba = y_logit[target_cls]
                cf_ctx_probas = self.context_classifier(z)
                #cf_sample = self.decoder(z)

                # Losses

                target_logit = tf.eye(2)[target_cls,:]
                target_logit = tf.reshape(target_logit, [1,2])
                current_L_val = self._binary_crossentropy(target_logit,y_logit_)
               # AE_loss = tf.norm(self.target_prob-proba)

                # AE_loss= float(AE_loss.numpy())

                current_L_context = self._context_loss(target_context, cf_ctx_probas)
                cf_ctx_pred = (cf_ctx_probas.numpy() >= 0.5).astype(int)
                cf_ctx_probas    = cf_ctx_probas.numpy().squeeze()

                currect_L_dist = tf.norm(x_0 - cf_sample,ord=2)


                # Saving the best CF so far
                if (proba>= best_proba) or (proba >= self.target_prob):
                    if (context_loss <= best_L_context + best_L_context*0.1): 
                        best_cf = cf_sample
                        best_proba = proba
                        best_y_pred_label = y_pred_label
                        best_L_dist = currect_L_dist
                        found_best = True
                        best_context = cf_ctx_probas
                        best_L_context = current_L_context

               
            
                epoch = epoch+1 

                if verbose>0:
                    print(f"epoch {epoch}/{self.epochs} Target {self.target_prob} Target cls {target_cls} Context target {target_context.numpy().squeeze()} Best {found_best} Alpha {self.alpha:.2f}")
                    print(f"Pred logit [{float(y_logit[0]):.2f},{float(y_logit[1]):.2f}] Pred context [{float(cf_ctx_probas[0]):.2f},{float(cf_ctx_probas[1]):.2f}]")
                    print(f"Dist. {currect_L_dist:.2f} Val loss {current_L_val:.2f}  Context loss {context_loss:.2f} Loss {loss:.2f} ")
                    print(f"Grad mean/abs {np.mean(grad)}/{np.mean(np.abs(grad))}  Grad min/max {np.min(grad):.2f}/{np.max(grad):.2f}")
                    print(f"Best {found_best}   Best prob. {best_proba:.2f}  Best dist. {best_L_dist:.2f} Best context [{float(best_context[0]):.2f},{float(best_context[1]):.2f}]\n")

            CFs.append(best_cf.numpy())
            CFs_y_.append(best_y_pred_label)
        
        return np.concatenate(CFs,axis=0),np.array(CFs_y_),X,pred_labels
    

    def test_z_y(self,z):
        return self.model_z_to_y.predict(z)

    @tf.function
    def _loss(self,x_0,z_k, target_class,target_context):
        """
        Loss function for CF generation

        PARAMS
            original_sample [tf.variable] --> The original sample before start searching
            z_k [tf.Variable]             --> The z point 
            target_clas [int]             --> Target class of the final counterfactual  
            target_context [tf.Variable]  --> Target context labels
            alpha  [float] [0,1]          --> Ratio of influence for classifier and context losses

        """
        logit =self.model_z_to_y(z_k) 
        proba = tf.gather(logit, indices=target_class, axis=1)
        cf_ctx_probas = self.context_classifier(z_k)
        x_ = self.decoder(z_k)
        target_logit = tf.eye(2)[target_class,:]
        target_logit = tf.reshape(target_logit, [1,2])

        L_val = self._binary_crossentropy(target_logit, logit)
        L_context = self._context_loss(target_context, cf_ctx_probas)
        L_dist = tf.norm(x_0 - x_, ord=2)  
        L_total = self.alpha*L_val + (1-self.alpha)*L_context + self.gamma*L_dist
        return L_total
    

    @classmethod
    def get_model_type(cls):
        return "CF" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CondLatentCF" # Aquí se puede indicar un ID que identifique el modelo

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 