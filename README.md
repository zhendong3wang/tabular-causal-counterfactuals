# CausalCACTUS: A Causally Consistent and Context-aware Framework for Counterfactual Explanations
Official implementation of: "CausalCACTUS: A Causally Consistent and Context-aware Framework for Counterfactual Explanations" (Under Review)

## Requirements
All experiments are evaluated on a Tesla V100 DGXS 32GB GPU and an Intel(R) Xeon(R) CPU E5-2698 40-Core Processor, with following dependencies:
- Python version: 3.9 (key libraries can be found in `requirements.txt`)
- Cuda version: 11.2 

### Installation
```
pip install -r requirements.txt
```

## Reproducing Results
Step-by-step commands to reproduce:
- Causal graph learning scripts can be found in the [`notebooks_rex`](./notebooks_rex/) folder.

- Training auto-encoders (AEs) and classififers for experiments
```
bash trainAEs.sh

bash trainClassifiers.sh
```

- Evaluation scripts are available in the [`notebooks_causal_cactus`](./notebooks_causal_cactus/) folder.

- Ablation study scripts can be found in the [`notebooks_causal_cactus`](./notebooks_causal_cactus/) folder.

- Result files are available in the [`results`](./results/) folder.


## Data
All the datasets are publicly available tabular datasets across different application domains, listed in the [`Data`](./Data) folder.
Three of these datasets focus on credit approval prediction (where the prediction target is a binary credit risk or loan outcome label): 
- Give Me Credit (Credit)
- German Credit (German)
- Home Equity Line of Credit (HELOC)


Additionally, we considered two datasets from other domains: 
- Adult, from income dataset (with the prediction target as income levels)
- Law, from education (where the outcome is bar passage for law school students).

These datasets range in sample size from 462 to 16,668 instances and include varying numbers of features (both categorical and numerical), which demonstrate the generalizability of our experiments.
