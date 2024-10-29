![scRCA logo](https://github.com/LMC0705/scRCA/blob/main/scRCA_log.png)

![Python Versions](https://img.shields.io/badge/python-3.6+-brightgreen.svg)

# scRCA: A Siamese Network-based Pipeline for Cell Type Annotation Using Imperfect scRNA-seq Reference Data

---

## 📜 Abstract
Accurate cell type annotation is a critical step in single-cell transcriptomic (scRNA-seq) data analysis, typically relying on comparisons with reference datasets. However, reference datasets are often imperfect due to various factors, including laboratory and methodological errors, which can lead to annotation inaccuracies. Current analysis pipelines for scRNA-seq data do not fully address these issues, creating a need for a robust computational pipeline that can effectively handle noisy reference datasets.

**scRCA** is a Siamese network-based pipeline designed to accurately annotate cell types, even with imperfect reference data. To enhance trustworthiness, scRCA includes an interpreter to explore the factors behind model predictions, and it also incorporates three noise-robust loss-based methods to improve annotation accuracy. Benchmarking experiments demonstrate that scRCA outperforms existing methods in accuracy and can overcome batch effects across various scRNA-seq techniques.

<img src="https://github.com/LMC0705/scRCA/blob/main/figure.png" alt="figure" width="600"/>


---

## 📦 Requirements

To install necessary packages, run:
```bash
pip install scanpy==1.7.2 torch==1.7.2 lime==0.1.1.36
```

---

## 🚀 Installation

### Option 1: Install via pip
```bash
pip install ./packages/scRCA-0.1.0.tar.gz
```

### Option 2: Direct Installation (Recommended)
Since scRCA is designed with a simple network architecture, it can be easily used by cloning the repository locally:
```bash
git clone https://github.com/LMC0705/scRCA.git
```
You can then directly annotate query cells using the quick start guide below.

---

## 📊 Datasets
The reference and query datasets are available in the `Data` folder, sourced from the [Immune Cell Dataset](https://www.tissueimmunecellatlas.org/). Additional benchmark datasets can be downloaded from [scrnaseqbenchmark on DockerHub](https://hub.docker.com/u/scrnaseqbenchmark).

---

## ⚡ Quick Start Guide

```python
import scanpy as sc
from Annotation import scRCA_annotate

# Load reference and query datasets
Ref_data_path = "./data/refer_data.h5ad"
Query_data_path = "./data/query_data.h5ad"

# Annotate the query dataset
query_data = scRCA_annotate(
    Ref_data_path, Query_data_path,
    learning_rate=0.01, n_epoch=20, noise_rate=0.45,
    forget_rate=0.35, pretrain_epochs=30
)
pred_cell_type = query_data.obs['predicted_cell_types']
```

### Parameter Descriptions:
- **`learning_rate`**: Learning rate for the optimizer.
- **`n_epoch`**: Number of epochs for main training.
- **`noise_rate`**: Expected noise level in the reference dataset; represents the proportion of mislabeled data.
- **`forget_rate`**: Controls the rate at which noisy labels are ignored.
- **`pretrain_epochs`**: Epochs for the initial pretraining phase.

These parameters can be adjusted within `scRCA_annotate`, providing flexibility to configure the annotation process.

---

## 📞 Contact
For questions or support, please contact: **008474@yzu.edu.cn**

--- 

This README provides a streamlined, easy-to-follow introduction to scRCA, helping researchers get started with single-cell data annotation using the power of Siamese networks and noise-robust learning.
