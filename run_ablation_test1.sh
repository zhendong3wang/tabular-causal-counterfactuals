#!/bin/bash

# conda activate py39

python cf_evaluate_ablation.py --dataset credit --prior min --ablation alpha --output results_credit_ablation1.csv
python cf_evaluate_ablation.py --dataset credit --prior mod --ablation alpha --output results_credit_ablation1.csv

python cf_evaluate_ablation.py --dataset adult --prior min --ablation alpha --output results_adult_ablation1.csv
python cf_evaluate_ablation.py --dataset adult --prior mod --ablation alpha --output results_adult_ablation1.csv

python cf_evaluate_ablation.py --dataset law --prior min --ablation alpha --output results_law_ablation1.csv
python cf_evaluate_ablation.py --dataset law --prior mod --ablation alpha --output results_law_ablation1.csv

python cf_evaluate_ablation.py --dataset germancredit --prior min --ablation alpha --output results_germancredit_ablation1.csv
python cf_evaluate_ablation.py --dataset germancredit --prior mod --ablation alpha --output results_germancredit_ablation1.csv

python cf_evaluate_ablation.py --dataset heloc --prior min --ablation alpha --output results_heloc_ablation1.csv
python cf_evaluate_ablation.py --dataset heloc --prior mod --ablation alpha --output results_heloc_ablation1.csv
