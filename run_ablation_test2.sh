#!/bin/bash

# conda activate py39

python cf_evaluate_ablation.py --dataset credit --prior min --ablation component --output results_credit_ablation2.csv
python cf_evaluate_ablation.py --dataset credit --prior mod --ablation component --output results_credit_ablation2.csv

python cf_evaluate_ablation.py --dataset adult --prior min --ablation component --output results_adult_ablation2.csv
python cf_evaluate_ablation.py --dataset adult --prior mod --ablation component --output results_adult_ablation2.csv

python cf_evaluate_ablation.py --dataset law --prior min --ablation component --output results_law_ablation2.csv
python cf_evaluate_ablation.py --dataset law --prior mod --ablation component --output results_law_ablation2.csv

python cf_evaluate_ablation.py --dataset germancredit --prior min --ablation component --output results_germancredit_ablation2.csv
python cf_evaluate_ablation.py --dataset germancredit --prior mod --ablation component --output results_germancredit_ablation2.csv

python cf_evaluate_ablation.py --dataset heloc --prior min --ablation component --output results_heloc_ablation2.csv
python cf_evaluate_ablation.py --dataset heloc --prior mod --ablation component --output results_heloc_ablation2.csv
