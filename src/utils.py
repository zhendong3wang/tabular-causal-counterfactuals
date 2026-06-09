# from scipy import signal
# from scipy.signal import medfilt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
# from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
import gc
from tensorflow.keras import backend as K
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import ipdb

def cleanup_gpu():
    # Delete model from memory
    K.clear_session()  # Clear backend session
    tf.keras.backend.clear_session()  # Clear Keras-related memory
    gc.collect()  # Run garbage collection

def upsampling(X,y, strategy="all"):
    oversample = RandomOverSampler(sampling_strategy=strategy)
    y_ = np.argmax(y,axis=1) if len(y.shape) > 1 else y 
    X_ = X[...,1] if len(X.shape)>2 else X
    oversample.fit_resample(X_,y)
    idx = oversample.sample_indices_
    return X[idx,...],y[idx,...], idx



# originally from: https://github.com/zhendong3wang/learning-time-series-counterfactuals/blob/main/src/help_functions.py
def euclidean_distance(X, cf_samples, average=True):
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))

    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances

def norm_euclidean_distance(X, cf_samples, average=True):
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))

    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    # Note: normalized by sqrt(#features)
    return np.mean(paired_distances)/np.sqrt(cf_samples.shape[1]) if average else paired_distances 


# originally from: https://github.com/zhendong3wang/learning-time-series-counterfactuals/blob/main/src/help_functions.py
def validity_score(pred_labels, cf_labels,accuracy_score):
    desired_labels = 1 - pred_labels  # for binary classification
    validity = accuracy_score(y_true=desired_labels, y_pred=cf_labels)
    return validity

# LOF of conunterfactuals on the true data distribution
def LOF_score(X_train, cf_samples):
    # Flattening the X and Xcf 
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    X_train     = np.reshape(X_train,(X_train.shape[0],-1))
    LOF_model =  LocalOutlierFactor(n_neighbors=int(np.sqrt(X_train.shape[0])),novelty=True)
    LOF_model.fit(X_train)
    LOF = LOF_model.score_samples(cf_samples)
    return np.mean(-LOF.squeeze())

# LOF of conunterfactuals on the true context distribution
def LOF_context_score(X_train, context_train, cf_samples, target_context, cat_idx=None, num_idx=None, agg=True):
    # Flattening the X and Xcf 
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    X_train     = np.reshape(X_train,(X_train.shape[0],-1))

    # One-hot for cat_idx if provided:
    if (cat_idx != None) and (num_idx != None):
        ct = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_idx),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
            ],
            remainder="drop"
        )
        # Fit on training data
        # ct.fit_transform(X_train) returns: scipy.sparse.csr_matrix; toarray() converts back to np.array
        X_train = ct.fit_transform(X_train)
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        cf_samples = ct.transform(cf_samples)
        if hasattr(cf_samples, "toarray"):
            cf_samples = cf_samples.toarray()

    LOFs = []
    LOFs_labels = np.unique(context_train,axis=0)
    # For each context class a LOF model is trained and all the Xcf belonging to this class are evaluated
    for context in LOFs_labels:
        context = context.tolist()

        #LOF value is only computed if there is samples of this context both in training dataset and in the generated Xcf
        if (np.sum(np.all(target_context == context, axis=1)) > 0) and (np.sum(np.all(context_train == context, axis=1)) > 20):
           
            # filtering X cf
            idx = np.all(target_context == context, axis=1)
            idx = np.where(idx)[0]
            cf_samples_ = cf_samples[idx]
                        
            # filtering X train
            idx = np.all(context_train == context, axis=1)
            idx = np.where(idx)[0]
            X_ = X_train[idx]

            #LOF_model =  LocalOutlierFactor(n_neighbors=int(np.sqrt(len(idx))),novelty=True)
            # n_neighbors = 10 # previous version
            n_neighbors = int(np.clip(np.sqrt(len(X_)), 20, 100)) # For small n → minimum 20 (stability) For medium n (~1000) → ~30 For large n (~3000) → ~55
            LOF_model =  LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, metric="euclidean")
            # TODO: "manhattan" or "euclidean"? -> "euclidean" could knock values down by several orders of magnitude
            
            # print("---- DEBUG Context-LOF score ----")
            # print("Context:", context)
            # print("Subset size:", X_.shape[0])
            # print("Unique rows:", np.unique(X_, axis=0).shape[0])

            X_ = np.unique(X_, axis=0)
            # print("After np.unique(): subset size:", X_.shape[0])
            
            LOF_model.fit(X_) # It returns a number. Large negative values means outlier; values close to 0 means inliers
            
            lof_cf = -LOF_model.score_samples(cf_samples_)
            lof_train = -LOF_model.score_samples(X_)

            # print("cf_samples_ shape:", cf_samples_.shape)
            # print("X_ shape:", X_.shape)
            # print("lof_cf min/max:", np.min(lof_cf), np.max(lof_cf))
            # print("lof_train min/max:", np.min(lof_train), np.max(lof_train))
            # print("mean train LOF:", np.mean(lof_train))

            LOFs.append(lof_cf)

    LOFs = np.hstack(LOFs).squeeze()
    # print(np.min(LOFs), np.median(LOFs), np.max(LOFs), len(LOFs))
    if agg:
        return np.mean(LOFs)
    else:
        return LOFs

# LOF of counterfactuals on the true context distribution (binary version)
def LOF_inliner_fraction(X_train, context_train, cf_samples, target_context,
                      cat_idx=None, num_idx=None, agg=True):

    # Flatten
    cf_samples = np.reshape(cf_samples, (cf_samples.shape[0], -1))
    X_train = np.reshape(X_train, (X_train.shape[0], -1))

    # --- One-hot encoding if needed ---
    if (cat_idx is not None) and (num_idx is not None):
        ct = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_idx),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_idx),
            ],
            remainder="drop"
        )
        X_train = ct.fit_transform(X_train)
        cf_samples = ct.transform(cf_samples)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(cf_samples, "toarray"):
            cf_samples = cf_samples.toarray()

    inlier_fractions = []
    LOF_labels = np.unique(context_train, axis=0)

    for context in LOF_labels:
        context = context.tolist()

        has_cf = np.sum(np.all(target_context == context, axis=1)) > 0
        enough_train = np.sum(np.all(context_train == context, axis=1)) > 20

        if not (has_cf and enough_train):
            continue

        # --- Filter CFs ---
        cf_idx = np.where(np.all(target_context == context, axis=1))[0]
        cf_samples_ = cf_samples[cf_idx]

        # --- Filter Training ---
        train_idx = np.where(np.all(context_train == context, axis=1))[0]
        X_ = X_train[train_idx]

        # Remove duplicates in training
        unique_before = X_.shape[0]
        X_ = np.unique(X_, axis=0)
        unique_after = X_.shape[0]

        # Safe neighbor selection
        # n_neighbors = 10 # previous version
        n_neighbors = int(np.clip(np.sqrt(len(X_)), 20, 100)) # For small n → minimum 20 (stability) For medium n (~1000) → ~30 For large n (~3000) → ~55
        if n_neighbors < 5:
            print(f"[WARNING] Context {context} skipped (too few samples).")
            continue

        LOF_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            metric="euclidean"
        )

        LOF_model.fit(X_)

        # --- Binary prediction ---
        preds = LOF_model.predict(cf_samples_)  # 1 = inlier, -1 = outlier
        n_inliers = np.sum(preds == 1)
        frac_inliers = n_inliers / len(preds)

        # # --- Debug prints ---
        # print("---- DEBUG Context-LOF (Binary) ----")
        # print("Context:", context)
        # print("Train size:", unique_after, f"(removed {unique_before - unique_after} duplicates)")
        # print("CF count:", len(cf_samples_))
        # print("n_neighbors:", n_neighbors)
        # print("Inliers:", n_inliers, "/", len(preds))
        # print("Fraction inliers:", round(frac_inliers, 4))
        # print("------------------------------------")

        inlier_fractions.append(frac_inliers)

    if len(inlier_fractions) == 0:
        print("[WARNING] No valid contexts evaluated.")
        return 0.0

    inlier_fractions = np.array(inlier_fractions)

    # print("==== FINAL CONTEXT-LOF SCORE ====")
    # print("Per-context fractions:", np.round(inlier_fractions, 4))
    # print("Mean fraction:", np.mean(inlier_fractions))
    # print("==================================")

    if agg:
        return float(np.mean(inlier_fractions))
    else:
        return inlier_fractions


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples, atol=1e-2, average=True):
    X           = np.reshape(X,(X.shape[0],-1))
    cf_samples  = np.reshape(cf_samples,(cf_samples.shape[0],-1))
    c_scores = np.isclose(X, cf_samples, atol=atol) 
    # Note: if average=True, then average per-sample and per-feature; eles, return raw matrix
    return np.mean(c_scores) if average else c_scores

# ---------- Causal CF evaluation helpers ----------
# same idea to causal validity defined in: https://github.com/tridungduong16/MultiObjective-SCM-CounterfactualExplanation/blob/36d85ef858e3e104565925836e04717afcb08782/src/evaluation_func.py#L348
def hard_causal_validity(X, CF_X, child_to_parents, atol=1e-5, average=True):
    causal_val_hard = []
    for x, x_cf in zip(X, CF_X):
        delta = np.abs(x_cf - x)
        # print(delta)
        
        ok = True
        for child, parents in child_to_parents.items():
            if delta[child] > atol:
                if not np.any(delta[parents] > atol):
                    ok = False
                    break
        causal_val_hard.append(ok)
    return np.mean(causal_val_hard) if average else causal_val_hard


def soft_causal_validity(X, CF_X, child_to_parents, atol=1e-5, average=True):
    causal_val_soft = []
    for x, x_cf in zip(X, CF_X):
        delta = np.abs(x_cf - x)
        per_constraint = []
        for child, parents in child_to_parents.items():
            if delta[child] <= atol:
                per_constraint.append(1.0)
            else:
                per_constraint.append(float(np.any(delta[parents] > atol)))
        causal_val_soft.append(np.mean(per_constraint))
    return np.mean(causal_val_soft) if average else causal_val_soft

def causally_compact_validity(X, CF_X, child_to_parents, atol=1e-5, comp_atol=1e-2, average=True):

    scv_list = soft_causal_validity(X, CF_X, child_to_parents, atol, average=False)
    comp_matrix = compactness_score(X, CF_X, comp_atol, average=False)
    comp_list = np.mean(comp_matrix, axis=1) # average to sample level
    scores = scv_list * comp_list

    return np.mean(scores) if average else scores


def cf_eval(
    X_train,
    context_train,
    X,
    CF_X,
    pred_labels, 
    target_context, 
    CF_labels, 
    X_unscaled,
    CF_X_unscaled,
    child_to_parents_dict=None, 
    cat_idx=None,
    num_idx=None,
    average=True,
    accuracy_score=accuracy_score
):

    dist        = euclidean_distance(X,CF_X)
    dist_norm   = norm_euclidean_distance(X,CF_X)
    validity    = validity_score(pred_labels,CF_labels,accuracy_score)
    compactness = compactness_score(X,CF_X)
    lof_context = LOF_context_score(X_train, context_train, CF_X, target_context, cat_idx, num_idx, agg=average)
    inlier_fraction = LOF_inliner_fraction(X_train, context_train, CF_X, target_context, cat_idx, num_idx, agg=average)
    #lof         = LOF_score(X_train, CF_X)  

    # # For causal evaluation helpers: scale back to original space
    # # X_unscaled = X
    # # CF_X_unscaled = CF_X
    causal_hard = hard_causal_validity(X_unscaled, CF_X_unscaled, child_to_parents_dict, average=average)
    causal_soft = soft_causal_validity(X_unscaled, CF_X_unscaled, child_to_parents_dict, average=average)
    causal_compact = causally_compact_validity(X_unscaled, CF_X_unscaled, child_to_parents_dict, average=average)

    metrics = [
        dist,
        dist_norm,
        validity,
        compactness,
        lof_context,
        causal_hard,
        causal_soft,
        causal_compact,
        inlier_fraction
        ]
    metric_names = [
        "proximity",
        "n_proximity",
        "validity",
        "compactness",
        "lof_context",
        "causal_validity_hard",
        "causal_validity_soft",
        "causal_compact_val",
        "inlier_fraction"
        ]
    return metrics, metric_names



def pairwise_l2_norm2(x, y, scope=None):
    
    size_x = tf.shape(x).numpy()[0]
    size_y = tf.shape(y).numpy()[0]
    xx = tf.expand_dims(x, -1)
    xx = tf.tile(xx, tf.constant([1, 1, 1,size_y]))

    yy = tf.expand_dims(y, -1)
    yy = tf.tile(yy, tf.constant([1, 1,1, size_x]))
    yy = tf.transpose(yy, perm=[2, 1, 0])

    diff = tf.math.subtract(xx, yy)
    square_diff = tf.square(diff)

    square_dist = tf.reduce_sum(square_diff, 1)

    return square_dist


def polynomial_decay(initial_lr, end_lr, max_epochs, power, current_epoch):
    """
    Polynomial learning rate decay.

    Parameters:
        initial_lr (float): Initial learning rate
        end_lr (float): Final learning rate after decay
        max_epochs (int): Total number of epochs
        power (float): Polynomial power (1.0 = linear decay)
        current_epoch (int): Current epoch

    Returns:
        float: Adjusted learning rate
    """
    if current_epoch > max_epochs:
        return end_lr
    decay = (1 - current_epoch / max_epochs) ** power
    lr = (initial_lr - end_lr) * decay + end_lr
    return lr


def styling_mpl_table(table,table_data,fontsize = 12):
    """
        Customize the appearance: remove vertical lines and lighten horizontal lines
    """
    
    colLabels = [cell.get_text().get_text() for (row, col), cell in table.get_celld().items() if row==0]
    n_cols = len(colLabels)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # lighter lines
        cell.set_edgecolor('lightgray')  # lighter color
        cell.set_height(0.1) 
        # Remove vertical borders inside the table
        cell.visible_edges = 'horizontal'

        # Remove the first horizontal line
        if row ==0:
            cell.visible_edges = 'open'
     # Format of the table
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for i, col in enumerate(colLabels):
        max_len = max([len(str(row[i])) if row[i] is not None else 0 for row in table_data] + [len(col)])
        scale = max_len / 8  
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(scale * 0.1)  # Escala proporcional al contenido

    # Optional: style header separately
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_fontsize(fontsize)
        cell.set_text_props(weight='bold')
        cell.set_edgecolor('gray')
   
                  
