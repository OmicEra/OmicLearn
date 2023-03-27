"""OmicLearn UI Texts."""

APP_TITLE = "OmicLearn — ML platform for omics datasets"
DISCLAIMER_TEXT = """
**⚠️ Warning:** It is possible to get artificially high or low performance because of technical and biological artifacts in the data.
While OmicLearn has the functionality to perform basic exploratory data analysis (EDA) such as PCA, 
it is not meant to substitute throughout data exploration but rather add a machine learning layer.
Please check our [recommendations](https://OmicLearn.readthedocs.io/en/latest/recommendations.html) 
page for potential pitfalls and interpret performance metrics accordingly.

**Note:** By uploading a file, you agree to our [Apache License](https://github.com/MannLabs/OmicLearn/blob/master/LICENSE.txt).
**Uploaded data will not be saved.**
"""

CITATION_TEXT = """**Reference:** 
Torun, F. M., Virreira Winter, S., Doll, S., Riese, F. M., Vorobyev, A., Mueller-Reif, J. B., Geyer, P. E., & Strauss, M. T. (2022).
Transparent Exploration of Machine Learning for Biomarker Discovery from Proteomics and Omics Data.
Journal of Proteome Research. https://doi.org/10.1021/acs.jproteome.2c00473"""

BUG_REPORT_TEXT = """We appreciate community contributions to the repository.
Here, you can [report a bug on GitHub](https://github.com/MannLabs/OmicLearn/issues/new/choose)."""

FILE_UPLOAD_TEXT = """Maximum size 200 MB. One row per sample, one column per feature. 
\nFeatures (proteins, genes, etc.) should be uppercase, all other additional features with a leading '_'.
"""

ALZHEIMER_CITATION_TEXT = """
**This dataset was retrieved from the following paper and the code for parsing is available at
[GitHub](https://github.com/MannLabs/OmicLearn/blob/master/data/Alzheimer_paper.ipynb):**\n
Bader, J., Geyer, P., Müller, J., Strauss, M., Koch, M., & Leypoldt, F. et al. (2020).
Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease.
Molecular Systems Biology, 16(6). doi: [10.15252/msb.20199356](http://doi.org/10.15252/msb.20199356)
"""

PACKAGES_PLAIN_TEXT = """
OmicLearn ({OmicLearn_version}) was utilized for performing data analysis, model execution, and creation of plots and charts.
Machine learning was done in Python ({python_version}). 
Feature tables were imported via the Pandas package ({pandas_version}) and manipulated using the Numpy package ({numpy_version}).
The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}).
The Plotly ({plotly_version}) library was used for plotting.
"""

SUBSET_TEXT = """This section allows you to specify a subset of data based on values within a comma.
Hence, you can exclude data that should not be used at all."""

CLASSIFICATION_TARGET_TEXT = """
Classification target refers to the column that contains the variables that are used two distinguish the two classes.
In the next section, the unique values of this column can be used to define the two classes.
"""

DEFINE_CLASS_TEXT = """
For a binary classification task, one needs to define two classes based on the
unique values in the **`{STATE_TARGET_COLUMN}`** task column.
It is possible to assign multiple values for each class.
"""

EXCLUDE_FEATURES_TEXT = """Exclude some features from the model training by selecting or uploading a CSV file.
This can be useful when, e.g., re-running a model without a top feature and assessing the difference in classification accuracy.
"""

ADDITIONAL_FEATURES_TEXT = "Select additional features. All non numerical values will be encoded (e.g. M/F -> 0,1)"

MANUALLY_SELECT_FEATURES_TEXT = "Manually select a subset of features. If only these features should be used, additionally set the "
"`Feature selection` method to `None`. "
"Otherwise, feature selection will be applied, and only a subset of the manually selected features is used. "
"Be aware of potential overfitting when manually selecting features and "
"check [recommendations](https://OmicLearn.readthedocs.io/en/latest/recommendations.html) page for potential pitfalls."


RESULTS_TABLE_INFO = """
**Info:** "Mean precision" and "Mean recall" values provided in the table above
are calculated as the mean of all individual splits shown in the confusion matrix,
not the "Sum of all splits" matrix.
"""

XGBOOST_NOT_INSTALLED = "**WARNING:** Xgboost not installed. To use xgboost install using `conda install py-xgboost`"

EDA_TEXT = """Use exploratory data anlysis on your dateset to identify potential correlations and biases.
For more information, please visit
[the dedicated ReadTheDocs page](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#exploratory-data-analysis-eda).
"""

RUNNING_INFO_TEXT = """
**Running info:**
- Using **Positive Class: {STATE_CLASS_0}** and **Negative Class: {STATE_CLASS_1}** targets.
- Using **{STATE_CLASSIFIER}** classifier.
- Using a total of **{LEN_STATE_FEATURES}** features.
- ⚠️ **Warning:** OmicLearn is intended to be an exploratory tool to assess the performance of algorithms,
    rather than providing a classification model for production. 
    Please check our [recommendations](https://OmicLearn.readthedocs.io/en/latest/recommendations.html)
    page for potential pitfalls and interpret performance metrics accordingly.
"""
