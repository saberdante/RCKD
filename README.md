# RCKD: Rethinking the Dark Knowledge and Kullback-Leibler Divergence Loss in Knowledge Distillation under Capacity Mismatching

This repository contains the code for the paper, "RCKD: Hierarchical Self-Distillation for Personalized Federated Learning".

We would like to express our gratitude to the authors of the [mdistiller](https://github.com/megvii-research/mdistiller) library, upon which our work is based.

## How to Run

### 1. Configure Environment:

```bash
pip install -r requirements.txt
python setup.py develop
```

### 2. Run:

```bash
python tools/train.py --cfg configs/cifar100/RCKD/res32x4_res8x4.yaml
```