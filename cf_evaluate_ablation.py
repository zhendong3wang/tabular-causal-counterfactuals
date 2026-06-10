#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

from src.DataReaders import *
from src.address import *
from src.modelGen import *
from src.utils import cf_eval, cleanup_gpu, MetricWriter
from models import CF_CausalCondLatentCF, CF_LatentCFpp, CF_PrototypeCF, CF_CondLatentCF 

import numpy as np
import pandas as pd
import random 
from tensorflow import random as tf_random
import tensorflow as tf

import argparse
import json
import re
from collections import defaultdict
import logging

SEED = 23

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf_random.set_seed(SEED)

CF_METHODS = {  
    "LatentCFpp": CF_LatentCFpp,
    "Prototype": CF_PrototypeCF,
    "CausalCACTUS": CF_CausalCondLatentCF,
    "CACTUS": CF_CondLatentCF
}

ABLATION_VARIANTS = {
    "baseline": {
        "CFmethod": "CACTUS",
        "dynamicAlpha": False,
        "use_phase_stopping": False,
        "use_clipping": False,
        "use_causal_graph": False
    },
    "causal": {
        "CFmethod": "CausalCACTUS",
        "dynamicAlpha": False,
        "use_phase_stopping": False,
        "use_clipping": True,
        "use_causal_graph": True
    },
    "phase": {
        "CFmethod": "CausalCACTUS",
        "dynamicAlpha": True,
        "use_phase_stopping": True,
        "use_clipping": False,
        "use_causal_graph": False
    },
    "full": {
        "CFmethod": "CausalCACTUS",
        "dynamicAlpha": True,
        "use_phase_stopping": True,
        "use_clipping": True,
        "use_causal_graph": True
    }
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["credit", "adult", "law", "germancredit", "heloc"],
        default="credit",
        help="Dataset name"
    )

    parser.add_argument(
        "--prior",
        choices=["min", "mod"],
        default="min",
        help="Causal prior graph"
    )

    parser.add_argument(
        "--output",
        default="results.csv"
    )

    parser.add_argument(
        "--ablation",
        choices=["alpha", "component"],
        required=True
    )

    A = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}.")

    # Generate the config for the experiment
    with open("./configs/experiments/datasets.json") as f:
        DATASETS_CONFIG = json.load(f)

    if A.ablation == "alpha":
        EXP = generate_config_ablation1(A.dataset, DATASETS_CONFIG)
        logger.info(EXP)
    elif A.ablation == "component":
        EXP = generate_config_ablation2(A.dataset, DATASETS_CONFIG)
        logger.info(EXP)
    else:
        logger.info("Not implemented.")
    
    # Load dataset causal graphs
    graph_path = f"./configs/causal_graphs_rex/{A.dataset}/"
    with open(os.path.join(graph_path, f"constraints_{A.prior}.dot"), 'r') as file:
        graph_str = file.read()

    writer = MetricWriter(
        A.output,
        A.dataset,
        A.prior
    )
    if not os.path.isfile(A.output):
        writer.write_head()
    
    # ## Graph 1 (Minimum priors) or Graph 2 (Moderate priors)
    N = 50 
    N_BOOTSTRAP = 5
    if A.prior == "min":
        SEED_list = [12, 23, 34, 45, 56]
    else:
        SEED_list = [11, 22, 33, 44, 55]

    # Number of samples to compute the metrics for CF evaluation
    for i, exp in enumerate(EXP):
        logger.info("\n" * 2)
        logger.info("*" * 200)
        logger.info(f"Running exp: {exp['name']} ({i}/{len(EXP)})")
        logger.info("*" * 200)
        logger.info("\n" * 2)
        
        # Reading the data
        CLASS_CONFIG_PATH = exp["classifier"]
        AE_CONFIG_PATH = exp["AEmodel"]

        class_config = get_exp_config(CLASS_CONFIG_PATH)
        logger.info("Reading data")
        data = getData(class_config)

        # Getting classifier
        classifier = modelGen(class_config["type"], data, params=class_config, debug=True)
        classifier.load()

        # Getting AE-based Model
        AE_config = get_exp_config(AE_CONFIG_PATH)
        aeModel = modelGen(AE_config["type"], data, params=AE_config, debug=True)
        aeModel.load()

        # CF generation
        child_to_parents_dict, parent_to_children_dict, feat2idx = build_causal_index_map(
            data.features_lbls,
            graph_str
        )
        logger.info("Feature → index:")
        logger.info(feat2idx)
        logger.info("\nChild → Parents (index form):")
        logger.info(child_to_parents_dict)
        
        if "CausalCACTUS" in exp["name"]:
            CF_method = CF_METHODS[exp["CFmethod"]](
                classifier=classifier,
                gen=aeModel,
                params=exp,
                x=data.X_train,
                y=data.y_train,
                x_min=data.X_train.min(axis=0),
                x_max=data.X_train.max(axis=0),
                causal_index_map=child_to_parents_dict
            )
        else:
            CF_method = CF_METHODS[exp["CFmethod"]](
                classifier=classifier,
                gen=aeModel,
                params=exp,
                x=data.X_train,
                y=data.y_train
            )

        for trial in range(N_BOOTSTRAP):

            # --- Sampling test data ---
            SEED = SEED_list[trial]
            os.environ['PYTHONHASHSEED'] = str(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            tf_random.set_seed(SEED)
            rand_idx = np.random.choice(len(data.X_test), N, replace=True)
            
            X_test = data.X_test[rand_idx, ...]
            context_test = data.context_test[exp['context']].values
            context_test = context_test[rand_idx, ...]

            context_training = data.context_train[exp['context']].values

            y_test = np.argmax(data.y_test[rand_idx, ...], axis=1)
            y_original_logits = classifier.predict(X_test)
            y_original_labels = np.argmax(y_original_logits, axis=1)

            # --- CF generation ---
            X_cf, y_cf_labels, X_internal, y_internal = CF_method.transform(
                X_test,
                y_original_labels,
                target_context=context_test,
                verbose=0
            )              
        
            # --- CF evaluation ---
            cf_scores, cf_scores_labels = cf_eval(
                data.X_train, 
                context_training, 
                X_test, 
                X_cf,  
                y_original_labels, 
                context_test, 
                y_cf_labels, 
                data.scaler_inverse_transform(X_test),
                data.scaler_inverse_transform(X_cf),
                child_to_parents_dict
            )

            writer.write_result(
                model_name=exp["name"],
                cf_method=exp["CFmethod"],
                bootstrap_id=trial,
                metric_results=cf_scores,
            )

        # Cleaning GPU models
        cleanup_gpu()

    logger.info("Done.")

def generate_config_ablation1(dataset, DATASETS_CONFIG):
    with open("./configs/experiments/ABLATION_EXP1.json") as f:
        ablation_config = json.load(f)

    EXP = []
    dataset_config = DATASETS_CONFIG[dataset]

    for method in ablation_config["methods"]:
        for alpha in ablation_config["alphas"]:
            exp = {
                "name": f"{method}-a{alpha:.2f}",
                "data": dataset,
                "classifier": dataset_config["classifier"],
                "AEmodel": dataset_config["AEmodel"],
                "context": dataset_config["context"],
                "CFmethod": method,
                "epochs": 100,
                "target_prob": 0.5,
                "learning_rate": 0.01,
                "power": 0.5,
                "alpha": alpha,
                "gamma": 0.1,
                "beta": 0.01,
                "dynamicAlpha": False,
                "use_phase_stopping": (
                    method == "CausalCACTUS"
                ),
                "use_clipping": (
                    method == "CausalCACTUS"
                ),
                "use_causal_graph": (
                    method == "CausalCACTUS"
                )
            }

            EXP.append(exp)
    return EXP


def generate_config_ablation2(dataset, DATASETS_CONFIG):
    with open("./configs/experiments/ABLATION_EXP2.json") as f:
        ablation_config = json.load(f)

    EXP = []
    dataset_cfg = DATASETS_CONFIG[dataset]

    for variant_name in ablation_config["variants"]:
        variant = ABLATION_VARIANTS[variant_name]

        EXP.append({
                "name": f"CausalCACTUS-{variant_name}",
                "data": dataset,
                "classifier": dataset_cfg["classifier"],
                "AEmodel": dataset_cfg["AEmodel"],
                "context": dataset_cfg["context"],
                "epochs": 300,
                "target_prob": 0.5,
                "learning_rate": 0.01,
                "power": 0.5,
                "alpha": 0.7,
                "gamma": 0.1,
                "beta": 0.01,
                **variant
            })
        
    return EXP

def build_causal_index_map(input_features, graph_str):
        """
        Returns:
            child_to_parents: dict {child_idx: [parent_idx, ...]}
            parent_to_children: dict {parent_idx: [child_idx, ...]}
            feat2idx: dict {feature_name: index}
        """

        # Feature name to index
        feat2idx = {f: i for i, f in enumerate(input_features)}

        # Extract edges using regex
        edges = re.findall(r'(\w+)\s*->\s*(\w+)', graph_str)

        child_to_parents = defaultdict(list)
        parent_to_children = defaultdict(list)

        for parent, child in edges:

            if parent not in feat2idx or child not in feat2idx:
                raise ValueError(f"Feature in graph not found in dataset: {parent} or {child}")

            p_idx = feat2idx[parent]
            c_idx = feat2idx[child]

            child_to_parents[c_idx].append(p_idx)
            parent_to_children[p_idx].append(c_idx)

        return dict(child_to_parents), dict(parent_to_children), feat2idx

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()