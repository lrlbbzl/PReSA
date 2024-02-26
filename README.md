## Prior Relational Schema Assists Effective Contrastive Learning for Inductive Knowledge Graph Completion

Code of COLING 2024 paper  Prior Relational Schema Assists Effective Contrastive Learning for Inductive Knowledge Graph Completion.

In this article, we propose an efficient KGE method based on relational schema and contrastive learning for inductive knowledge graph completion tasks.

## Requirements
* python>=3.7
* torch>=1.6
* transformers>=4.15

We have run the experiments on 4 * A40 (when using high batch size).

## Tutorial

### Humanswiki-ind

* Training

  ```bash
  bash scripts/train_hw.sh
  ```

  Modify parameters such as `CUDA_VISIBLE_DEVICES` according to your own needs.

* Inference

  ```bash
  bash scripts/eval.sh ./checkpoint/humans_wikidata_ind/model_last.mdl humans_wikidata_ind
  ```

## Acknowledge

Thanks the sharing from [SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models - ACL Anthology](https://aclanthology.org/2022.acl-long.295/)

