<p align="left"><img src="https://github.com/LMC0705/scRCA/blob/main/scRCA_log.png" width="250" height="100"></p>

![Python Versions](https://img.shields.io/badge/python-3.6+-brightgreen.svg)

A Siamese network-based pipeline for the annotation of cell types using imperfect single-cell RNA-seq reference data

# Abstract
A critical step in the analysis of single-cell transcriptomic (scRNA-seq) data is the accurate identification and annotation of cell types. Such annotation is usually conducted by comparative analysis with known (reference) data sets – which assumes an accurate representation of cell types within the reference sample. However, this assumption is often incorrect, because factors, such as human errors in the laboratory or in silico, and methodological limitations, can ultimately lead to annotation errors in a reference dataset. As current pipelines for single-cell transcriptomic analysis do not adequately consider this challenge, there is a major demand for a computational pipeline that achieves high-quality cell type annotation using imperfect reference datasets that contain inherent errors (often referred to as “noise”). Here, we built a Siamese network-based pipeline, termed scRCA, that achieves an accurate annotation of cell types employing imperfect reference data. For researchers to decide whether to trust the scRCA annotations, an interpreter was developed to explore the factors on which the scRCA model makes its predictions. We also implemented 3 noise-robust losses-based cell type methods to improve the accuracy using imperfect dataset. Benchmarking experiments showed that scRCA outperforms the proposed noise-robust loss-based methods and methods commonly in use for cell type annotation using imperfect reference data. Importantly, we demonstrate that scRCA can overcome batch effects induced by distinctive single cell RNA-seq techniques. 
![image](https://github.com/LMC0705/scRCA/blob/main/figure.png)

# Requirement:
```console
pip install scanpy=1.7.2
pip install torch=1.7.2
pip install lime=0.1.1.36
```
# Install scRCA
### Using pip 
```console
pip install ./packages/scRCA-0.1.0.tar.gz
```

# Datasets
The reference dataset and query dataset contained in Data are derived from Immune Cell Dataset(https://www.tissueimmunecellatlas.org/)]
The benchmark datasets can be downloaded from https://hub.docker.com/u/scrnaseqbenchmark

#Use
```console
import scanpy as sc
import 
####################load data###############
refer=sc.read_h5ad("./data/refer_data.h5ad")
refer_data=refer.X
refer_label=refer.obs["celltype"]

query=sc.read_h5ad("./data/query_data.h5ad")
query_data=query.X
query_label=query.obs["celltype"]

##########annotatie the query dataset#########################
predict_label=scRCA.annotate(refer_data,refer_label,query_data)
```

# Contact
Please contact us if you have any questions: liuyan@njust.edu.cn
