#!/bin/bash

CONFIGS=(
    "exp/GIVECREDIT_AE/config.json" \
    "exp/GIVECREDIT_CACTUS/config.json" \
    "exp/ADULT_AE/config.json" \
    "exp/ADULT_CACTUS/config.json" \
    "exp/LAW_AE/config.json" \
    "exp/LAW_CACTUS/config.json" \
    "exp/GERMANCREDIT_AE/config.json" \
    "exp/GERMANCREDIT_CACTUS/config.json" \
    "exp/HELOC_AE/config.json" \
    "exp/HELOC_CACTUS/config.json" 
)


echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    CUDA_VISIBLE_DEVICES=3 python train.py --config $CONFIG
done



