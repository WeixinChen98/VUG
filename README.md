

# VUG

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)



**VUG** is the official implementation for paper "Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users".


## Requirements

Before running the code, make sure the following dependencies are installed:

```
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Quick-Start Guide

To begin using the VUG library, use the following command:

```bash
python run_recbole_cdr.py --model=[model] --config_file=[config_file]
```

### Example: Running VUG on the Epinions Dataset

```bash
python run_recbole_cdr.py --model=VUG --config_file=./recbole_cdr/properties/dataset/Epinions.yaml
```

## Key Files and Modules

To help users understand the structure of the repository and locate the implementation of key components, here is a breakdown of important files:

- **Model File**:  
  Located in: `./recbole_cdr/model/cross_domain_recommender/vug.py`  
  This file contains the implementation of the proposed VUG model.

- **Loss Function**:  
  Located in: `./recbole_cdr/model/cross_domain_recommender/utils.py`  
  This file implements the constraint loss.

- **Attention-Based Generator**:  
  Located in: `./recbole_cdr/model/cross_domain_recommender/attention.py`  
  This file contains the attention-based generator model used in VUG.

## Configuration and Model Selection

The configuration files define the dataset and hyperparameter settings. Example configurations can be found in the `./recbole_cdr/properties/dataset/` directory.  

### Example Config File: `Epinions.yaml`

```yaml
field_separator: ","
source_domain:
  dataset: EpinionsElec
  data_path: '/xxx/VUG/recbole_cdr/dataset'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[1,inf)"
  item_inter_num_interval: "[1,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

target_domain:
  dataset: EpinionsGame
  data_path: '/xxx/VUG/recbole_cdr/dataset'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[1,inf)"
  item_inter_num_interval: "[1,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True


epochs: 500
train_batch_size: 4096
eval_batch_size: 409600000
valid_metric: NDCG@10
```

To customize experiments, modify the configuration file or provide additional parameters via command-line arguments.

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR). 



