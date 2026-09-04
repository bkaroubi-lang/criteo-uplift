---
license: cc-by-nc-sa-4.0
tags:
- criteo
- advertising
- causality
pretty_name: criteo-uplift
size_categories:
- 10M<n<100M
task_categories:
- tabular-classification
---

# Introduction

This dataset is released along with the paper:

[A Large Scale Benchmark for Uplift Modeling](https://openreview.net/pdf?id=Q83-QeTB9lS)
Eustache Diemert, Artem Betlei, Christophe Renaudin; (Criteo AI Lab), Massih-Reza Amini (LIG, Grenoble INP)

This work was published in: AdKDD 2018 Workshop, in conjunction with KDD 2018.

When using this dataset, please cite the paper with following bibtex:

    @inproceedings{Diemert2018,
      author = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
      title={A Large Scale Benchmark for Uplift Modeling},
      publisher = {ACM},
      booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, London,United Kingdom, August, 20, 2018},
      year = {2018}
    }


# Data description

This dataset is constructed by assembling data resulting from several incrementality tests, a particular randomized trial procedure where a random part of the population is prevented from being targeted by advertising. it consists of 25M rows, each one representing a user with 11 features, a treatment indicator and 2 labels (visits and conversions).

## Fields

Here is a detailed description of the fields (they are comma-separated in the file):

- f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11: feature values (dense, float)
- treatment: treatment group (1 = treated, 0 = control)
- conversion: whether a conversion occured for this user (binary, label)
- visit: whether a visit occured for this user (binary, label)
- exposure: treatment effect, whether the user has been effectively exposed (binary)


## Privacy

For privacy reasons the data has been sub-sampled non-uniformly so that the original incrementality level cannot be deduced from the dataset while preserving a realistic, challenging benchmark. Feature names have been anonymized and their values randomly projected so as to keep predictive power while making it practically impossible to recover the original features or user context.

## Key figures

Format: CSV
Size: 297M (compressed)
Rows: 13,979,592
Average Visit Rate: .046992
Average Conversion Rate: .00292
Treatment Ratio: .85

# Tasks and Code

The dataset can be used primarily to benchmark methods in Uplift Modeling, Individual Treatment Effect prediction / Heterogeneous Treatment Effect.

Reference paper: [ITE and UM](https://openreview.net/pdf?id=Q83-QeTB9lS)

Reference experimental code and evaluation: [Github](https://github.com/criteo-research/large-scale-ITE-UM-benchmark)