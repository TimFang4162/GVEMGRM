# GVEMGRM: An R Package for Variational Estimation in the Multidimensional Graded Response Model

## Introduction
GVEMGRM is an R package that implements the Gaussian Variational Expectation Maximization (GVEM) algorithm for estimating item parameters in the Multidimensional Graded Response Model (MGRM). It supports both confirmatory and exploratory analysis of Likert-scale response data with high-dimensional latent traits. Additionally, an importance-weighted GVEM (IW-GVEM) algorithm is included to reduce bias in the estimation of the loading matrix during confirmatory analysis.

## Installation
To install the package, download the source file GVEMGRM_0.0.tar.gz and run the following command in R:

``` r
install.packages("PATH_TO_FILE/GVEMGRM_0.0.tar.gz", repos = NULL, type = "source")
```

Replace PATH_TO_FILE with the actual path to the downloaded file.


## Package Contents

The package provides three core functions and one example dataset:

- *gvemgrm*: Implements the GVEM algorithm for MGRM.
- *iwgvemgrm*: Implements the IW-GVEM algorithm for MGRM.
- *simudata*: Generates synthetic datasets for simulation studies.
- *toy_data*: A toy dataset for demonstrating the use of the algorithms.

For detailed usage, function arguments and examples, please refer to the help pages via ?function_name in R (e.g., ?gvemgrm).

## Citation
If you use this package in your research or publications, please cite the following work:

Zheng, Q. Z., Shang, L., Xu, P. F.*, Shan, N., & Gao, Z. (2025). Variational Estimation for Multidimensional Graded Response Model.

Available at: https://github.com/Laixu3107/GVEMGRM
