#!/bin/bash

CONFIGS=(
    "configs/models/GIVEMECREDIT_class/config.json" \
    "configs/models/ADULT_class/config.json" \
    "configs/models/LAW_class/config.json" \
    "configs/models/GERMANCREDIT_class/config.json" \
    "configs/models/HELOC_class/config.json" 
    )

echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    CUDA_VISIBLE_DEVICES=3 python train.py --config $CONFIG
done



