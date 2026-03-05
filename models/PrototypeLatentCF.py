
import ipdb
import tensorflow as tf
from address import *
layers = tf.keras.layers
K = tf.keras.backend
initialiazers = tf.keras.initializers
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanAbsoluteError,BinaryCrossentropy
from .CNN import CNN
from .BaseModel import BaseModel
import numpy as np
import pandas as pd
from itertools import product
import sys
sys.path.insert(0, os.path.abspath('../'))
from utils import pairwise_l2_norm2,polynomial_decay

class PrototypeLatentCF():
    """
        Counterfactual generator inspired in:

        Van Looveren, A., & Klaise, J. (2021, September). Interpretable counterfactual explanations guided by prototypes. 
        In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 650-665).
        Cham: Springer International Publishing.
            - https://github.com/SeldonIO/alibi
        

    """
    def __init__(self, classifier:CNN, gen:BaseModel, params: dict, x, y, **kwargs)->None:

        # Prototypes CF params 
        self.epochs            = params.get('epochs',30) 
        self.tol               = params.get('tol',0.01)
        self.target_prob       = params.get('target_prob',0.9)
        self.kappa             = params.get('kappa',0)
        self.c_steps           = params.get('c_steps',5)
        self.c                 = params.get('c',1)       # For L_pred
        self.beta              = params.get('beta',0.1 ) # For L_1
        self.gamma             = params.get('gamma',100) # For L_AE
        self.theta             = params.get('theta',100) # For L_proto
        self.clip              = params.get("clip",(-1000,1000)) # Grad boundaries
        self.feat_range        = params.get("feat_range",(0,1))  # Input feat range
        self.learning_rate     = params.get("lr",1e-2)
        self.poly_decay_power  = params.get("poly_decay_power",0.5)
        
        # Models 
        self.classifierObj   =  classifier
        self.classifier        = self.classifierObj.model
        self.nb_cls            = self.classifierObj.nb_clss
        self.genModel          =  gen        
        self.decoder           = self.genModel.model_dec
        self.encoder           = self.genModel.model_enc
        self.z_input           = self.genModel.z_input

        # Input sample to be explained
        self.target_prob       = tf.constant([self.target_prob]) 
        self.x                 = x
        self.y                 = y
        

        self.loss_history      = []
        # Building the CF generation model
        self.fit(self.x,self.y)

    def _getClassPrototypes(self,z,y):

        """
        It gets the prototypes by averaging all the encoded samples of each class

        PARAMS

        z [numpy array]           ->  Encoded samples from which the prototype is computed
        y [numpy array]           ->  labels for computing the prototype
        """
        proto = []        
        lbls = self.y.shape[1]
        for lbl in range(lbls):
            proto.append(np.mean(z[y[:,lbl]==1],axis=0,keepdims=True))
        return proto



    def fit(self,x,y):
        """
            
            It computes the prototypes of each class.

            PARAMS

            x [numpy array]           -> Example of input to compute the context prototypes.
            y [numpy array]          -> Class labels to compute the prototypes.
        """
        print(f"Building model: {PrototypeLatentCF.get_model_name()} [{PrototypeLatentCF.get_model_type()}]")
        
        # Getting the centroids for each cotext class
        z = self.encoder(x).numpy()
        self.prototypes = self._getClassPrototypes(z,y)
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


    def transform(self,X,pred_labels,target_context,verbose=1):
        """
            Generates the CF. The initial sample x is iteratively transformed to minimize the target loss: 
                Loss = c*L_pred + L_dist + + L_AE + L_proto  

            The opt is repeated c_steps times adjusting c according to the cf obtained. c  is increased or decreased, respectively,
            depending on whether CF is found.
            
            PARAMETERS
                X [numpy array]                 -> Input sample to be transformed
                pred_labels [numpy array]       -> Estimated labels for X
                target_context [2D numpy array] -> Labels of the target context 
            RETURN
                CF [numpy array]                -> Estimated counterfactuals
                CFs_y_ [numpy arrya]            -> Estimated labels for the counterfactual samples
        """
        N_samples = X.shape[0]
        CFs=[]
        CFs_y_ = []
        context = target_context # Not used in prototype
        Loss_info = []    
        self.GRADS = []   

        for i_sample in range(N_samples):

            x               = X[[i_sample]]
            y               = pred_labels[i_sample]

            # TF variables on which the opt. will be performed 
            x_0             = tf.constant(x,dtype="float32",name='x0')
            x_delta         = tf.Variable(x,dtype="float32",name ='x_delta')
            x_delta_prev    = tf.Variable(x,dtype="float32",name ='x_delta_prev')
            
            # Selecting the best proto. for sample x
            self.proto           = self._get_best_proto(x,y)
            target_cls      = 1 - y
            
            
            ## Initializing vars needed to select best CF
            best_dist   = 10e20
            best_cf     = np.zeros_like(x)
            best_y_     = 0
            find_best   = False
            c = self.c
            c_steps = self.c_steps
            
            while (c_steps > 0):
                epoch =0
                while (self.epochs > epoch):

                    # Registering the gradient of loss wrt x_delta
                    with tf.GradientTape() as tape:
                        tape.watch(x_delta)
                        loss = self._loss(x_0,x_delta,self.proto,target_cls,c)
                    
                    # Getting the gradient 
                    grad = tape.gradient(loss,x_delta)
                
                    # Clipping the gradients
                    grad = tf.where(grad < self.clip[0], self.clip[0], grad)
                    grad = tf.where(grad > self.clip[1], self.clip[1], grad)
                    self.GRADS.append(grad.numpy())

                    #Update of x_delta.
                    lr = polynomial_decay(self.learning_rate, 0, self.epochs, self.poly_decay_power, epoch)
                    y = x_delta - lr*grad
                    delta  = y -x_0

                    # Proximal of FISTA (Shrinkage thresholding)
                    delta = tf.where(tf.abs(delta) < self.beta, 0.0, delta) 
                    delta = tf.where(delta > self.beta, delta - self.beta, delta) 
                    delta = tf.where(delta < -self.beta, delta + self.beta, delta) 

                    y = x_0 + delta


                    # Nesterov momentum
                    x_delta = y + (epoch/(epoch+3))* ( y-x_delta_prev)

                    x_delta = tf.where(tf.abs(x_delta) < self.beta, 0, x_delta)
                    x_delta = tf.where(x_delta < self.feat_range[0], self.feat_range[0], x_delta)
                    x_delta = tf.where(x_delta > self.feat_range[1], self.feat_range[1], x_delta)


                    x_delta_prev = y

                    y_logit         = self.classifier(x_delta)
                    y_              = np.argmax(y_logit.numpy().squeeze())
                    # Pred loss 
                    L_pred = self.__pred_loss(y_logit[:,1-target_cls],y_logit[:,target_cls],c)
                    L_pred= float(L_pred.numpy())
                    # Distance loss 
                    L_dist = self._dist_loss(x_0,x_delta)
                    L_dist= float(L_dist.numpy())
                    # AE loss 
                    L_AE   = self._recons_loss(x_delta)
                    L_AE= float(L_AE.numpy())
                    # Proto Loss 
                    L_proto = self._proto_loss(x_delta,self.proto)
                    L_proto= float(L_proto.numpy())
                    
                    # Update the best cf
                    if y_ == target_cls:
                        if L_dist < best_dist:
                            best_dist = L_dist
                            best_cf = x_delta.numpy()
                            best_y_ = y_
                            find_best = True
                            # print("New best CF found!")

                    # Showing the opt information
                    loss = float(loss)
                    if verbose>0:
                        print(f"Epoch {epoch}/{self.epochs}   Target cls {target_cls}  ")
                        print(f"Pred logit: [{y_logit[0,0]:.2f},{y_logit[0,1]:.2f}] Best proto: {find_best}  c: {c} ")
                        print(f"L_pred: {L_pred:.2f}  L_dist: {L_dist:.2f}  L_AE: {L_AE:.2f}  L_proto: {L_proto:.2f}  Loss: {loss:.2f} ")
                        print(f"Grad abs/mean: {np.mean(np.abs(grad.numpy())):.2f}/{np.mean(grad.numpy()):.2f}  Grad min/max: {np.min(grad.numpy()):.2f}/{np.max(grad.numpy()):.2f} \n")
                    
                    # Gathering opt. history
                    Loss_info.append([i_sample,L_pred,L_dist,L_AE,L_proto,loss])
                    
                    epoch = epoch+1 


                # Updating c
                if (find_best):
                    c = c/2.
                else: 
                    c = c*10.0

                # Reinitializing the opt vars
                x_delta       = tf.Variable(x,dtype="float32",name ='x_delta')
                x_delta_prev  = tf.Variable(x,dtype="float32",name ='x_delta')
                find_best     = False
                #best_dist     = 10e10
                c_steps = c_steps-1 

            CFs.append(best_cf)
            CFs_y_.append(best_y_)
        Loss_info = pd.DataFrame(Loss_info,columns = ["sample","L_pred [c]","L_dist [beta]","L_AE [gamma]","L_proto [theta]","loss"])
        
        self.loss_history = Loss_info
        return np.vstack(CFs),np.array(CFs_y_),X,pred_labels
    

    def test_z_y(self,z):
        return self.model_z_to_y.predict(z)
    
    @tf.function
    def _loss(self,x,x_delta,proto,target,c):
        """
        Loss function for prototype CF generation

        """
        y_logit         = self.classifier(x_delta)
        delta           = x_delta -x 
        # Pred loss 
        L_pred = self.__pred_loss(y_logit[:,1-target],y_logit[:,target],c)
        # Distance loss (only L2 for opt, L1 is minimized by proximal (FISTA))
        L_dist = tf.reduce_sum(tf.square(delta))
        # AE loss 
        L_AE   = self._recons_loss(x_delta)
        # Proto Loss (x_delta, proto)
        L_proto = self._proto_loss(x_delta,proto)

        return  L_pred + L_dist + L_AE + L_proto

    def _get_best_proto(self,x,y):
        candidates = self.prototypes[:y] + self.prototypes[y+1:]
        z = self.encoder(x)
        dist = []
        for candidate in candidates:
            dist.append(tf.norm(z-candidate,ord=2).numpy())
        idx_best = np.argmin(np.array(dist))
        return candidates[idx_best]
    
    @tf.function
    def __pred_loss(self,y,y_delta,c):
        return tf.reduce_sum(tf.multiply(c, tf.math.maximum(y-y_delta+self.kappa, 0)))
    
    @tf.function
    def _dist_loss(self,x,x_delta):
        return tf.square(tf.norm(x-x_delta)) + self.beta*tf.norm(x_delta-x, ord=1)
    
    @tf.function
    def _recons_loss(self,x_delta):
        x_recons = self.genModel.model(x_delta)
        return self.gamma*tf.square(tf.norm(x_delta-x_recons))

    @tf.function
    def _proto_loss(self,x_delta,proto):
        return self.theta*tf.square(tf.norm(self.encoder(x_delta)-proto))

    @classmethod
    def get_model_type(cls):
        return "CF" 
    
    @classmethod
    def get_model_name(cls):
        return "PrototypeLatentCF" 

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 