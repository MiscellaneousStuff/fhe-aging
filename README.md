# FHE Aging

## About

Original: [Implement an FHE-based Biological Age and Aging Pace Estimation ML Model Using Zama Libraries](https://github.com/zama-ai/bounty-program/issues/143)

## Plan of Attack

1. Assess the accuracy and computational complexity of different approaches
2. When we've found an aging model which has a good balance, then start porting to FHE
   - This part might be reciprocal, will have to see what ends up being easy / hard to port.
     However, in theory that assessment should be pretty deterministic based on Big O and operator
     characteristics from the chosen model.

## Optimisation Characteristics

- Start with simpler models

<details>
<summar>Challenge Data</summary>

### Datasets

- The Illumina HumanMethylation450 BeadChip data
- GEO datasets like GSE40279 (often used for Horvath's clock)
- TCGA (The Cancer Genome Atlas) methylation data

#### `dnaMethyAge` R Package - Datasets

```
27k_reference: probeAnnotation21kdatMethUsed
CBL_common: coefs
CBL_specific: coefs
Cortex_common: coefs
DunedinPACE: coefs gold_standard_means
HannumG2013: coefs
HorvathS2013: coefs
HorvathS2018: coefs
LevineM2018: coefs
LuA2019: coefs
McEwenL2019: coefs
ShirebyG2020: coefs
YangZ2016: epiTOCcpgs
ZhangQ2019: coefs
ZhangY2017: coefs
subGSE174422: betas info
```

`betas`: Mthylation beta values - actual DNA methylation measurements that serve as input
features for the model.

`coefs`: Coefficient matrices for different biological clock models.
Each named entry represents a different published biological age clock with its trained
coefficients.

`probeAnnotation21kdatMethUsed`: Annotation data for DNA methlyation probes (CpG sites)
used in the models.
</details>