#!/bin/bash

# conda activate py39

python cf_evaluate_main.py --dataset credit --prior min --output results_credit.csv
python cf_evaluate_main.py --dataset credit --prior mod --output results_credit.csv

python cf_evaluate_main.py --dataset adult --prior min --output results_adult.csv
python cf_evaluate_main.py --dataset adult --prior mod --output results_adult.csv

python cf_evaluate_main.py --dataset law --prior min --output results_law.csv
python cf_evaluate_main.py --dataset law --prior mod --output results_law.csv

python cf_evaluate_main.py --dataset germancredit --prior min --output results_germancredit.csv
python cf_evaluate_main.py --dataset germancredit --prior mod --output results_germancredit.csv

python cf_evaluate_main.py --dataset heloc --prior min --output results_heloc.csv
python cf_evaluate_main.py --dataset heloc --prior mod --output results_heloc.csv

