# FHE Aging

## About

Zama.ai Bounty for Season 8: [Implement an FHE-based Biological Age and Aging Pace Estimation ML Model Using Zama Libraries](https://github.com/zama-ai/bounty-program/issues/143)

## Context

[Human arrays](`/pyaging/tutorials/tutorial_dnam_illumina_human_array.ipynb`)

## Age Prediction Context

1. Models trained to predict chronological age of tissue based on biomarkers
2. Delta betwen chronological age and real age used as marker to predict
   - Mortality risk
   - Disease states
   - etc.

## Glossary

- CpG group := (Cytosine - phosphate - Guanine)
  - Notable for role in gene regulation through methylation processes

### Inputs and Outputs for Age Prediction / Age Pacing

- Inputs (CgPs)
- Outputs (Predicted chronological age)

### Datasets and Models

| Dataset | Model |
| - | - |
| Horvath | ElasticNet |
| AltumAge | Deep Learning-based |
| PCGrimAge | PCA based version of GrimAge |
| GrimAge2 | Latest version of GrimAge ? |
| DunedinPACE | Biomarker of the pace of aging |
  
## Plan of Attack

### 1. Test Different Models + Assess Performance

1. Start with simpler models (linear regression-based clocks) as easier to implement in FHE
2. Balance accuracy vs compute complexity - some models might use hundreds of CpG sites that would be expensive
in FHE
3. Horvath clock well established but uses elastic net regression with many features
4. DunedinPACE measures aging pace rather than biological age, which might be interesting but more complex

### 2. Balance FHE Implementation Feasibility

1. Linear models most straightforward
2. Avoid non-linear activation functions, if possible
3. Consider feature count - Less CpG sites means faster FHE computation
4. Some biological clocks use relatively few CpG sites (~10-50) which would be ideal (NOTE: Need to validate this)

### 3. Port to Zama.ai's FHE Libraries

1. Start with Concrete ML for higher-level abstractions
2. Need to quantise model (using `brevitas-nn` / `concrete-ml`, etc.)
3. Benchmark acc between original + FHE - expect precision loss (though this depends on implementation / model, etc.)
4. Reduce multiplicative depth

### 4. Optimise for Efficiency

- Use concrete's compiler to analyse circuit depth / bottlenecks
- Precision vs performance trade-off (this one is likely key)
- Heavily consider preprocessing strategies pre-encryption to offload computation

### 5. Deploy to HuggingFace Spaces

1. Client: Encrypts methylation data
2. Server: Processes encrypted data without decryption
3. Client: Receives and decrypts the predicted biological age
- Demo + sample data

<details>
<summary>Number of Features per Dataset (for `pyaging`)</summary>

</details>

<details>
<summary>Challenge Data</summary>

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

`betas`: Methylation beta values - actual DNA methylation measurements that serve as input
features for the model.
`X`

`coefs`: Coefficient matrices for different biological clock models.
Each named entry represents a different published biological age clock with its trained
coefficients.
`Weights?`

`probeAnnotation21kdatMethUsed`: Annotation data for DNA methlyation probes (CpG sites)
used in the models.
</details>