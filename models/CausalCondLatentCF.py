
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


class CausalCondLatentCF():
    """
        Counterfactual generator using a conditioned Latent Space

    """
    def __init__(self, classifier:CNN, gen:BaseModel, causal_index_map, params: dict, **kwargs)->None:

        # CF params and models 
        #self.optimizer         = params.get('optimizer',tf.optimizers.Adam(learning_rate=0.2))
        
        self.epochs            = params.get('epochs',30)
        self.tol               = params.get('tol',0.01)
        self.target_prob       = params.get('target_prob',0.5)
        self.alpha             = params.get('alpha',0.5)        # "alpha": 0.7,
        self.beta              = params.get('beta',0.1)        # "beta": 0.01,
        self.learning_rate     = params.get('learning_rate',0.01)        # "learning_rate": 0.01,
        self.power             = params.get('power',2)        # "power": 0.5,
        self.gamma             = params.get('gamma',1)        # "gamma": 0.1,
        self.loss              = params.get("loss",self._loss)
        self.dynamicAlpha      = params.get("dynamicAlpha",False)

        # ===== ABLATION FLAGS =====
        self.use_phase_stopping = params.get("use_phase_stopping", True)
        self.use_clipping = params.get("use_clipping", True)
        self.use_causal_graph = params.get("use_causal_graph", True)
        
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
                
        self.causal_index_map = causal_index_map
        # === FEATURE RANGE CONSTRAINTS ===
        self.x_min = tf.constant(kwargs["x_min"], dtype=tf.float32)
        self.x_max = tf.constant(kwargs["x_max"], dtype=tf.float32)
        
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
        print(f"Building model: {CausalCondLatentCF.get_model_name()} [{CausalCondLatentCF.get_model_type()}]")
        #ipdb.set_trace()

        # Note: explictly freeze pretrained parts
        self.decoder.trainable = False
        self.classifier.trainable = False
        for layer in self.decoder.layers:
            layer.trainable = False 
        for layer in self.classifier.layers:
            layer.trainable = False

        self.model_z_to_y = Model(inputs=[self.z_input], outputs = [self.classifier(self.decoder(self.z_input))])
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

            best_proba = 0.3 # TODO: best_proba = 1 (?)
            best_y_pred_label = y_pred_label
            found_best = False
            found_valid_cf = False
            best_L_dist = 1e10
            best_context = cf_ctx_probas
            best_L_context = 1e10
            best_L_val = 1e10
            best_L_total = 1e10

            # TODO: add as a parameter
            patience = 50 
            no_improve_count = 0

            # Note: self.dynamicAlpha (and self.use_phase_stopping) will override self.alpha at each phase
            if self.dynamicAlpha and self.use_phase_stopping:
                self.alpha = 0.99
            
            # W_causal_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=tf.float32)
            # W_causal_mask = tf.constant([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=tf.float32)     
            W_causal_mask = tf.ones_like(x_0)
            delta_eps = 1e-3
            
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
                    loss = self._loss(x_0, z, target_cls, target_context, W_causal_mask)

                # Getting and applying the gradient wrt z
                grad = tape.gradient(loss,z)
                z = z - self.learning_rate*grad/tf.reduce_max(tf.abs(grad))
                # z = z - self.learning_rate*grad # debug
                # self.optimizer.apply_gradients([(grad, z)]) # debug
                

                cf_sample = self.decoder(z)
                if self.use_clipping:
                    # ======== FEATURE RANGE CLIPPING ========
                    cf_sample = tf.clip_by_value(cf_sample, self.x_min, self.x_max)

                # note: proximal L1 sparsity step, like elastic net CF methods
                # note: self.beta = 0.01
                # FISTA proximal (L1 opt.)
                delta = cf_sample - x_0
                delta = tf.where(tf.abs(delta) < self.beta, 0, delta)
                cf_sample = x_0 + delta
                if self.use_clipping:
                    # ======== RE-CLIP AFTER SPARSITY ========
                    cf_sample = tf.clip_by_value(cf_sample, self.x_min, self.x_max)

                if self.use_causal_graph:
                    # ===== CAUSAL DEPENDENCY GRAPH =====
                    # Note: update the causal mask here, w.r.t delta
                    delta = tf.abs(cf_sample - x_0)
                    delta = tf.squeeze(delta)
                    changed = tf.cast(delta > delta_eps, tf.float32)

                    # TODO: make this child_to_parents as a function input
                    # child_to_parents = {2: [6, 4], 4: [0, 3]} 
                    # print(f"child_to_parents: {self.causal_index_map}")
                    for child, parents in self.causal_index_map.items():
                        # print(f"W_causal_mask before: {W_causal_mask}")
                    
                        # print(delta)
                        # print(changed)
                        # print(f"child: {child}")
                        # print(f"parents: {parents}")

                        parent_changed = tf.reduce_max(
                            tf.gather(changed, parents)
                        )
                        child_changed = changed[child]

                        # print(f"parent_changed: {parent_changed}")
                        # print(f"child_changed: {child_changed}")

                        # Step 1: parent triggers child movement
                        W_child = tf.where(
                            parent_changed > 0,
                            0.01, # weight for the child when parent feature chagnes
                            1.0
                        )

                        # Step 2: once child moved, restore normal weight
                        W_child = tf.where(
                            child_changed > 0,
                            1.0,
                            W_child
                        )

                        W_causal_mask = tf.tensor_scatter_nd_update(
                            W_causal_mask, 
                            [[0, child]],       # [[row 0, column]] = index of the child
                            [W_child]
                        )
                        # print(f"W_causal_mask after: {W_causal_mask}")

                # Computing prediction and decoding z
                #z = tf.Variable(self.encoder(cf_sample))
                y_logit_ = self.model_z_to_y(z)
                y_logit  = y_logit_.numpy().squeeze()
                y_pred_label = np.argmax(y_logit)
                proba = y_logit[target_cls] # TODO: in the pseudo code: |y_cf − ŷ_k| (prediction error)

                cf_ctx_probas = self.context_classifier(z)
                #cf_sample = self.decoder(z)

                ###################################
                # Losses for logging
                ###################################
                # Compute raw losses (without scaling)
                target_logit = tf.eye(2)[target_cls,:]
                target_logit = tf.reshape(target_logit, [1,2])
                current_L_val = self._binary_crossentropy(target_logit, y_logit_)
                # in Glacier: current_L_val = tf.norm(self.target_prob - proba)
                
                current_L_context = self._context_loss(target_context, cf_ctx_probas)
                cf_ctx_pred = (cf_ctx_probas.numpy() >= 0.5).astype(int)
                cf_ctx_probas    = cf_ctx_probas.numpy().squeeze()

                # currect_L_dist = tf.norm(W_causal_mask * (x_0 - cf_sample), ord=1) 
                currect_L_dist = tf.norm(W_causal_mask * (x_0 - cf_sample), ord=2) 

                ###################################
                # Saving the best CF so far
                ###################################
                if self.use_phase_stopping:
                    improved_flag = False
                    # --- phase 1: validity milestone ---
                    # print(f"proba: {proba}")
                    if proba >= self.target_prob and (not found_valid_cf):
                        best_cf = cf_sample
                        best_proba = proba
                        best_y_pred_label = y_pred_label
                        best_context = cf_ctx_probas

                        best_L_val = current_L_val
                        best_L_dist = currect_L_dist
                        best_L_context = current_L_context
                        best_L_total = loss

                        found_valid_cf = True
                        improved_flag = True

                        # Note: self.dynamicAlpha will override self.alpha at each phase
                        # (Note: not penalize L_context if not changing the original context)
                        if self.dynamicAlpha:
                            self.alpha = 0.7
                    
                    # --- phase 2: context and loss milestone (must also valid) ---
                    # print("While-loop condition: option 3")
                    context_match = np.all(cf_ctx_pred == target_context.numpy().astype(int))
                    if proba >= self.target_prob and (context_match or current_L_context <= best_L_context):
                        # if loss <= best_L_total: # Option 1: optimizing for the total loss
                        # if currect_L_dist <= best_L_dist: # Option 2: optimizing for the distance loss
                        if loss <= best_L_total and currect_L_dist <= best_L_dist: # Option 3
                            best_cf = cf_sample
                            best_proba = proba
                            best_y_pred_label = y_pred_label
                            best_context = cf_ctx_probas

                            best_L_val = current_L_val
                            best_L_dist = currect_L_dist
                            best_L_context = current_L_context
                            best_L_total = loss

                            found_best = True
                            improved_flag = True
                            # Note: self.dynamicAlpha will override self.alpha at each phase
                            if self.dynamicAlpha: 
                                self.alpha = 0.6

                    # if improved_flag: 
                    #     no_improve_count = 0  # reset patience
                    # else:
                    #     no_improve_count += 1
                    # if no_improve_count >= patience:
                    #     if verbose > 0:
                    #         print(f"Early stopping at epoch {epoch}: no improvement for {patience} iterations.")
                    #     break       
                else:
                    # vanilla CACTUS: saving the best CF so far
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
                    print(f"i_sample/N_samples: {i_sample}/{N_samples}")
                    print(f"#epoch {epoch}/{self.epochs}.") # #no_improve_count/patience: {no_improve_count}/{patience}")
                    print(f"self.alpha: {self.alpha}. self.gamma: {self.gamma}") # for debugging
                    print(f"Valid {found_valid_cf}. Best {found_best}.")
                    print(f"Pred logits: [{float(y_logit[0]):.2f},{float(y_logit[1]):.2f}] Target prob {self.target_prob} Target cls {target_cls} ")
                    print(f"Pred context: [{float(cf_ctx_probas[0]):.2f},{float(cf_ctx_probas[1]):.2f}] Target ctx {target_context}")
                    print(f"Raw Dist loss {currect_L_dist:.2f} Val loss {current_L_val:.2f}  Context loss {current_L_context:.2f} ")
                    print(f"Scaled Dist loss {self.gamma*currect_L_dist:.2f} Val loss {self.alpha*current_L_val:.2f}  Context loss {(1-self.alpha)*current_L_context:.2f}; Loss total {loss:.2f} ")
                    print(f"Grad mean/abs {np.mean(grad):.2f}/{np.mean(np.abs(grad)):.2f}  Grad min/max {np.min(grad):.2f}/{np.max(grad):.2f}")
                    print(f"Best prob. {best_proba:.2f} Best val. {best_L_val:.2f} Best dist. {best_L_dist:.2f} Best context {best_L_context:.2f}\n")

            CFs.append(best_cf.numpy())
            CFs_y_.append(best_y_pred_label)
        
        return np.concatenate(CFs,axis=0),np.array(CFs_y_),X,pred_labels
    

    def test_z_y(self,z):
        return self.model_z_to_y.predict(z)

    @tf.function
    def _loss(self, x_0, z_k, target_class, target_context, W_causal_mask):
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
        # proba = tf.gather(logit, indices=target_class, axis=1)
        cf_ctx_probas = self.context_classifier(z_k)
        x_ = self.decoder(z_k)

        if self.use_clipping:
            # ======== CLIP INSIDE LOSS ========
            x_ = tf.clip_by_value(x_, self.x_min, self.x_max)

        target_logit = tf.eye(2)[target_class,:]
        target_logit = tf.reshape(target_logit, [1,2])

        L_val = self._binary_crossentropy(target_logit, logit)
        L_context = self._context_loss(target_context, cf_ctx_probas)
        # TODO: add a anti_proximity/L_causal term?
        """
        # L_causal = tf.norm(W_causal_mask * (x_z - x_), ord=2)) 
        # L_dist = tf.norm((x_0 - x_), ord=2) # + L_causal
        """
        # L_dist = tf.norm(W_causal_mask * (x_0 - x_), ord=1)  
        # in Glacier: tf.reduce_mean(tf.abs(x_-x_0))
        if self.use_causal_graph:
            L_dist = tf.norm(W_causal_mask * (x_0 - x_), ord=2)  
        else:
            L_dist = tf.norm(x_0 - x_, ord=2)
        L_total = self.alpha*L_val + (1-self.alpha)*L_context + self.gamma*L_dist
        return L_total
    

    @classmethod
    def get_model_type(cls):
        return "CF" # Aquí se puede indicar qué tipo de modelo es: RRNN, keras, scikit-lear, etc.
    
    @classmethod
    def get_model_name(cls):
        return "CausalCondLatentCF" # Aquí se puede indicar un ID que identifique el modelo

    @classmethod
    def is_model_for(cls,name):
        return cls.get_model_name() == name 