As of 11-03-2025.

## Overview

At 45 years old, an individual has completed 45 rotations around the sun. However, their biological age may differ from their chronological age. Biological age refers to the age of their cells, which may or may not align with their chronological age. People age at different rates, and what is most important is their healthspan—the period of life free from aging-related diseases.

In the past, researchers believed that genetics were seen as a key factor determining an individual's healthspan. Recent studies, however, suggest that genes account for only 15% to 25% of the aging process.

Biological age reflects the internal aging process of the body's cells, based on various factors and biomarkers that provide a snapshot of your current cellular health. Similarly, aging pace reflects how quickly this health span changes and can be estimated with tests like DunedinPACE.

Many biological clocks have been proposed in literature, and they work with varying types of biomarkers. Some examples are:

* [Steve Horvath](https://pubmed.ncbi.nlm.nih.gov/?term=Horvath+S&cauthor_id=24138928): [DNA methylation age of human tissues and cell types](https://pubmed.ncbi.nlm.nih.gov/24138928/)
* Levine et al: [An epigenetic biomarker of aging for lifespan and healthspan](https://pubmed.ncbi.nlm.nih.gov/29676998/)
* Belsky et al, DunedinPACE: [A DNA methylation biomarker of the pace of aging](https://elifesciences.org/articles/73420)
* Chen et al, OMICmAge: [An integrative multi-omics approach to quantify biological age with electronic medical records](https://www.biorxiv.org/content/10.1101/2023.10.16.562114v2)
* Sehgal et al, Systems Age: [A single blood methylation test to quantify aging heterogeneity across 11 physiological systems](https://www.biorxiv.org/content/10.1101/2023.07.13.548904v1.full.pdf)
* The following R library implements many biological clocks based on DNA methylation data: https://github.com/yiluyucheng/dnaMethyAge

## What we expect

* **Implement** biological age estimation machine learning models using Zama libraries, such as Concrete ML, Concrete, TFHE-rs or fhEVM.
* **Deploy** these models on encrypted data as Hugging Face spaces
* **Demonstrate** these models on sample data and demonstrate the models on encrypted data using FHE

### Judging criteria

Your solution should use Zama libraries such as Concrete ML, Concrete, TFHE-rs or fhEVM to implement an **FHE-based Biological Age and Aging Pace Estimation**. Key considerations include:

* **FHE implementation**: Choosing FHE-compatible biological age/aging pace estimation and implementing it efficiently.
* **Performance:** An evaluation of speed vs accuracy trade-offs when converting algorithms to use FHE.
* **Pre-processing**: Implementing pre-processing steps on clear data before applying FHE techniques.

We expect your submission to contain:

* **A report** detailing the method and technical choices you made, and evaluation of any tradeoffs that the implementation makes to achieve good accuracy and latency
* **Sample data** on which the models are demonstrated
* **A client/server application** on Hugging Face demonstrating your model on some data (which can be taken from a dataset) such as the one in the dnaMethyAge R package

## Reward

### 🥇Best submission: up to €5,000.

To be considered best submission, a solution must be efficient, effective and demonstrate a deep understanding of the core problem. Alongside the technical correctness, it should also be submitted with a clean code, clear explanations and a complete documentation.

### 🥈Second-best submission: up to €3,000.

For a solution to be considered the second best submission, it should be both efficient and effective. The code should be neat and readable, while its documentation might not be as exhaustive as the best submission, it should cover the key aspects of the solution.

### 🥉Third-best submission: up to €2,000.

The third best submission is one that presents a solution that effectively tackles the challenge at hand, even if it may have certain areas of improvement in terms of efficiency or depth of understanding. Documentation should be present, covering the essential components of the solution.

Reward amounts are decided based on code quality, model accuracy scores and speed performance on a m6i.metal AWS server. When multiple solutions of comparable scope are submitted they are compared based on the accuracy metrics and computation times.

## Related links and references

* [Pyaging Python package](https://github.com/rsinghlab/pyaging)
* [dnaMethyAge R package](https://github.com/yiluyucheng/dnaMethyAge)

## 👉 Register

### Step 1: Registration

Click [here](https://www.zama.ai/bounty-program/register) to register for the Zama Bounty Program. Fill out the registration form with your information. Once you fill out the form, you will receive a confirmation email with a link to the submission portal for when you are ready to submit your code.

Note

Check your spam folder in case you don't receive the confirmation email. If you haven't received it within 24 hour, please contact us by email at [bounty@zama.ai](mailto:bounty@zama.ai).

### Step 2: Work on the Challenge

Read through the Bounty details and requirements carefully. Use the provided resources and create your own GitHub repository to store your code.

If you have any questions during your work, feel free to comment directly in the Bounty issue and our team will be happy to assist you.

### Step 3: Submission

Once you have completed your work, upload your completed work to the submission portal using the link provided in the confirmation email.

Note

The deadline for submission is **May 18th, 2025** (Midnight, Anywhere On Earth). Late submissions will not be considered.

We wish you the best of luck with the challenge!

## ✅ Support

* Comment on this issue with any questions regarding this bounty.
* Email for private questions: [bounty@zama.ai](mailto:bounty@zama.ai).
* Join the Zama community channels [here](https://zama.ai/community).