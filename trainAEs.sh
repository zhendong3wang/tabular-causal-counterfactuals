#!/bin/bash

CONFIGS=(
    "configs/models/GIVEMECREDIT_AE/config.json" \
    "configs/models/GIVEMECREDIT_CACTUS/config.json" \
    "configs/models/ADULT_AE/config.json" \
    "configs/models/ADULT_CACTUS/config.json" \
    "configs/models/LAW_AE/config.json" \
    "configs/models/LAW_CACTUS/config.json" \
    "configs/models/GERMANCREDIT_AE/config.json" \
    "configs/models/GERMANCREDIT_CACTUS/config.json" \
    "configs/models/HELOC_AE/config.json" \
    "configs/models/HELOC_CACTUS/config.json" 
)


echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    CUDA_VISIBLE_DEVICES=3 python train.py --config $CONFIG
done



