

# VUG

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)



**VUG** is the official implementation for paper "Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users".


## Requirements

```
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Quick-Start Guide

To begin using our library, simply run the provided script with your desired model and configuration file:

```bash
python run_recbole_cdr.py --model=[model] --config_file=[config_file]
```

### Example

Run the VUG model on the Epinions dataset:

```bash
python run_recbole_cdr.py --model=VUG --config_file=./recbole_cdr/properties/dataset/Epinions.yaml
```

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR). 
