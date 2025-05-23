# Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices
This repository contains the code of the paper _Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices_ presented at ICML 2025. 

Thibault de Surrel, Fabien Lotte, Sylvain Chevallier and Florian Yger.

## Structure of this repository
- Source code: In the file `source`, you will find the source code for sampling a wrapped Gaussian on the manifold of SPD matrices given the parameters $(p, \mu, \Sigma)$. You will also find the code to estimate the parameters of an unknown wrapped Gaussian from a finite number of samples. 
- Experiments on the estimation: in the file `expe_estimation`, you will find the code used to run the experiments on the estimation of the parameters of a wrapped Gaussian. These codes were used to generate the results given at section 5 of the paper. 
- Experiments on the classification: in the file `expe_classification`, you will find the code used to test and compare the different classifiers we consider in the paper, at section 6. For some experiments, the datasets were too big for GitHub, so they are stored on Google Drive and need to be downloaded beforehand. 

## How to cite 
```latex
@inproceedings{deSurrelwrappedGaussians2025,
  author    = {de Surrel, Thibault and Lotte, Fabien and Chevallier, Sylvain and Yger, Florian},
  title     = {Wrapped Gaussian on the manifold of Symmetric Positive Definite Matrices},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
}
```
