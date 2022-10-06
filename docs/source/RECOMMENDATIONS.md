# Recommendations

This page gives an overview of recommendations to avoid potential pitfalls and misleading performance values.

## Potential pitfalls

### Artifacts
It is possible to get artificially high or low performance because of technical and biological artifacts in the data. This could be, for example, that samples from one group were measured or acquired differently than another group. While OmicLearn has the functionality to perform basic exploratory data analysis (EDA) such as PCA, it is not meant to substitute throughout data exploration but rather add a machine learning layer. Therefore, you should make sure that you don't have known biological or technical artifacts.

### Misuse of settings
Although OmicLearn tries to prevent invalid selections, it is sometimes possible, even by design. One of these is ‘Manually select features’. This feature is intended to explore how a classifier performs when a defined set of features is provided. However, it is to note that when extracting optimized features on the same dataset that will be used for training, e.g., by selecting proteins that already have been tested and found to be regulated, one is prone to get biased results, as previously described in
[How (Not) to Generate a Highly Predictive Biomarker Panel Using Machine Learning](https://pubs.acs.org/doi/10.1021/acs.jproteome.2c00117). This effect will not occur in the default settings as the feature selection will be applied on each cross-validation split.

### Sample Size
The sample size for ML analysis is a crucial parameter to get meaningful performance metrics. In the OmicLearn manuscript, we motivate the use of ML as proteomics reaches hundreds to thousands of samples, which should be seen as a minimum number.
In principle, we already risk overfitting in an ML context when we have more features than samples, which could easily be the case for a study with hundreds of patients and thousands of detected proteins.
Another aspect is that the reported metrics can become unreliable. This can be exemplified with some example calculations.
Consider the case of 100 samples: When using the default CV split of 5, we train on 80 samples and validate on 20. Only one misclassification would lead to 19 of 20 samples being correct and achieving an accuracy of 95% - so a change of 5%.

### Interpretation of results
Users should be advised that the performance values are reported on the cross-validation of training and validation data and not on the holdout set. In order to evaluate the performance of an ML model, a study needs to be split into a train, validation, and holdout (test) set. Optimization is performed using the training and validation set, and the model that is ultimately used is being evaluated using the unseen holdout set. OmicLearn is intended to be an exploratory tool to assess the performance of algorithms when applied to specific datasets at hand rather than a classification model for production. Therefore, no holdout set is used, and the performance metrics have to be interpreted accordingly. This also prevents repeated analysis of the same dataset and choosing the same holdout set from leading to a selection bias and consequent over-interpretation of the model.

## Other best practices
We can recommend the following guideline for Machine Learning in Proteomics:
 - [Interpretation of the DOME Recommendations for Machine Learning in Proteomics and Metabolomics](https://pubs.acs.org/doi/10.1021/acs.jproteome.1c00900)

## Studies using OmicLearn
To provide a perspective on the utility of OmicLearn, check out the following studies that used OmicLearn.

### Proteomics
- [High-resolution serum proteome trajectories in COVID-19 reveal patient-specific seroconversion](https://www.embopress.org/doi/full/10.15252/emmm.202114167)
- [Proteome profiling of cerebrospinal fluid reveals biomarker candidates for Parkinson’s disease](https://doi.org/10.1016/j.xcrm.2022.100661)

### Transcriptomics
- [Convergent cerebrospinal fluid proteomes and metabolic ontologies in humans and animal models of Rett syndrome](https://doi.org/10.1016/j.isci.2022.104966)


## 
Feel free to extend this list by suggesting additional recommendations.
