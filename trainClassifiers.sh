#!/bin/bash

CONFIGS=(
    "exp/GIVECREDIT_class/config.json" \
    "exp/ADULT_class/config.json" \
    "exp/LAW_class/config.json" \
    "exp/GERMANCREDIT_class/config.json" \
    "exp/HELOC_class/config.json" 
    )

echo "Starting training"

for CONFIG in "${CONFIGS[@]}"; do
    echo "training $CONFIG"
    CUDA_VISIBLE_DEVICES=3 python train.py --config $CONFIG
done



