# scRCA
A robust cell type annotation algorithm for single-cell RNA sequencing data
# Abstract
A critical step in the analysis of single-cell transcriptomic (scRNA-seq) data is the accurate identification and annotation of cell types. Such annotation is usually conducted by comparative analysis with known (reference) data sets – which assumes an accurate representation of cell types within the reference sample. However, this assumption is often incorrect, because factors, such as human errors in the laboratory or in silico, and methodological limitations, can ultimately lead to annotation errors in a reference dataset. As current pipelines for single-cell transcriptomic analysis do not adequately consider this challenge, there is a major demand for a computational pipeline that achieves high-quality cell type annotation using imperfect reference datasets that contain inherent errors (often referred to as “noise”). Here, we built a Siamese network-based pipeline, termed scRCA, that achieves an accurate annotation of cell types employing imperfect reference data. For researchers to decide whether to trust the scRCA annotations, an interpreter was developed to explore the factors on which the scRCA model makes its predictions. We also implemented 3 noise-robust losses-based cell type methods to improve the accuracy using imperfect dataset. Benchmarking experiments showed that scRCA outperforms the proposed noise-robust loss-based methods and methods commonly in use for cell type annotation using imperfect reference data. Importantly, we demonstrate that scRCA can overcome batch effects induced by distinctive single cell RNA-seq techniques. 
![image](https://github.com/LMC0705/scRCA/blob/main/figure.png)

Requirement:

scanpy 1.7.2

scikit-learn 0.24.2

torch 1.10.0

python 3.6.13

# Contact
Please contact us if you have any questions: liuyan@njust.edu.cn
