import plotly
import os, sys
import base64
import sklearn
import numpy as np
import pandas as pd
import streamlit as st

# Checkpoint for XGBoost
xgboost_installed = False
try:
    import xgboost
    from xgboost import XGBClassifier

    xgboost_installed = True
except ModuleNotFoundError:
    pass
# Widget for recording
def make_recording_widget(f, widget_values):
    """
    Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """

    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)

# Object for dict
class objdict(dict):
    """
    Objdict class to conveniently store a state
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


# Main components
def main_components():
    """
    Expose external CSS and create & return widgets
    """

    # Fundemental elements
    widget_values = objdict()
    record_widgets = objdict()

    # Sidebar widgets
    sidebar_elements = {
        "button_": st.sidebar.button,
        "slider_": st.sidebar.slider,
        "number_input_": st.sidebar.number_input,
        "selectbox_": st.sidebar.selectbox,
        "multiselect": st.multiselect,
    }
    for sidebar_key, sidebar_value in sidebar_elements.items():
        record_widgets[sidebar_key] = make_recording_widget(
            sidebar_value, widget_values
        )

    return widget_values, record_widgets


# Generate sidebar elements
def generate_sidebar_elements(state, icon, report, record_widgets):
    slider_ = record_widgets.slider_
    selectbox_ = record_widgets.selectbox_
    number_input_ = record_widgets.number_input_

    # Sidebar -- Image/Title
    st.sidebar.image(
        icon, use_column_width=True, caption="OmicLearn " + report["omic_learn_version"]
    )
    st.sidebar.markdown(
        "# [Options](https://omiclearn.readthedocs.io/en/latest//METHODS)"
    )

    # Sidebar -- Random State
    state["random_state"] = slider_(
        "Random State:", min_value=0, max_value=99, value=23
    )

    # Sidebar -- Preprocessing
    st.sidebar.markdown(
        "## [Preprocessing](https://omiclearn.readthedocs.io/en/latest//METHODS-%7C-1.-Preprocessing)"
    )
    normalizations = [
        "None",
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "PowerTransformer",
        "QuantileTransformer",
    ]
    state["normalization"] = selectbox_("Normalization method:", normalizations)

    normalization_params = {}

    if state.normalization == "PowerTransformer":
        normalization_params["method"] = selectbox_(
            "Power transformation method:", ["Yeo-Johnson", "Box-Cox"]
        ).lower()
    elif state.normalization == "QuantileTransformer":
        normalization_params["random_state"] = state.random_state
        normalization_params["n_quantiles"] = number_input_(
            "Number of quantiles:", value=100, min_value=1, max_value=2000
        )
        normalization_params["output_distribution"] = selectbox_(
            "Output distribution method:", ["Uniform", "Normal"]
        ).lower()
    if state.n_missing > 0:
        st.sidebar.markdown(
            "## [Missing value imputation](https://omiclearn.readthedocs.io/en/latest//METHODS-%7C-1.-Preprocessing#1-2-imputation-of-missing-values)"
        )
        missing_values = ["Zero", "Mean", "Median", "KNNImputer", "None"]
        state["missing_value"] = selectbox_("Missing value imputation", missing_values)
    else:
        state["missing_value"] = "None"

    state["normalization_params"] = normalization_params

    # Sidebar -- Feature Selection
    st.sidebar.markdown(
        "## [Feature selection](https://omiclearn.readthedocs.io/en/latest//METHODS-%7C-2.-Feature-selection)"
    )
    feature_methods = [
        "ExtraTrees",
        "k-best (mutual_info_classif)",
        "k-best (f_classif)",
        "k-best (chi2)",
        "None",
    ]
    state["feature_method"] = selectbox_("Feature selection method:", feature_methods)

    if state.feature_method != "None":
        state["max_features"] = number_input_(
            "Maximum number of features:", value=20, min_value=1, max_value=2000
        )
    else:
        # Define `max_features` as 0 if `feature_method` is `None`
        state["max_features"] = 0

    if state.feature_method == "ExtraTrees":
        state["n_trees"] = number_input_(
            "Number of trees in the forest:", value=100, min_value=1, max_value=2000
        )
    else:
        state["n_trees"] = 0

    # Sidebar -- Classification method selection
    st.sidebar.markdown(
        "## [Classification](https://omiclearn.readthedocs.io/en/latest//METHODS-%7C-4.-Classification#3-classification)"
    )
    classifiers = [
        "AdaBoost",
        "LogisticRegression",
        "KNeighborsClassifier",
        "RandomForest",
        "DecisionTree",
        "LinearSVC",
    ]
    if xgboost_installed:
        classifiers += ["XGBoost"]

    # Disable all other classification methods
    if (state.n_missing > 0) and (state.missing_value == "None"):
        classifiers = ["XGBoost"]

    state["classifier"] = selectbox_("Specify the classifier:", classifiers)
    classifier_params = {}
    classifier_params["random_state"] = state["random_state"]

    if state.classifier == "AdaBoost":
        classifier_params["n_estimators"] = number_input_(
            "Number of estimators:", value=100, min_value=1, max_value=2000
        )
        classifier_params["learning_rate"] = number_input_(
            "Learning rate:", value=1.0, min_value=0.001, max_value=100.0
        )

    elif state.classifier == "KNeighborsClassifier":
        classifier_params["n_neighbors"] = number_input_(
            "Number of neighbors:", value=100, min_value=1, max_value=2000
        )
        classifier_params["weights"] = selectbox_(
            "Select weight function used:", ["uniform", "distance"]
        )
        classifier_params["algorithm"] = selectbox_(
            "Algorithm for computing the neighbors:",
            ["auto", "ball_tree", "kd_tree", "brute"],
        )

    elif state.classifier == "LogisticRegression":
        classifier_params["penalty"] = selectbox_(
            "Specify norm in the penalization:", ["l2", "l1", "ElasticNet", "None"]
        ).lower()
        classifier_params["solver"] = selectbox_(
            "Select the algorithm for optimization:",
            ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
        )
        classifier_params["max_iter"] = number_input_(
            "Maximum number of iteration:", value=100, min_value=1, max_value=2000
        )
        classifier_params["C"] = number_input_(
            "C parameter:", value=1, min_value=1, max_value=100
        )

    elif state.classifier == "RandomForest":
        classifier_params["n_estimators"] = number_input_(
            "Number of estimators:", value=100, min_value=1, max_value=2000
        )
        classifier_params["criterion"] = selectbox_(
            "Function for measure the quality:", ["gini", "entropy"]
        )
        classifier_params["max_features"] = selectbox_(
            "Number of max. features:", ["auto", "int", "sqrt", "log2"]
        )
        if classifier_params["max_features"] == "int":
            classifier_params["max_features"] = number_input_(
                "Number of max. features:", value=5, min_value=1, max_value=100
            )

    elif state.classifier == "DecisionTree":
        classifier_params["criterion"] = selectbox_(
            "Function for measure the quality:", ["gini", "entropy"]
        )
        classifier_params["max_features"] = selectbox_(
            "Number of max. features:", ["auto", "int", "sqrt", "log2"]
        )
        if classifier_params["max_features"] == "int":
            classifier_params["max_features"] = number_input_(
                "Number of max. features:", value=5, min_value=1, max_value=100
            )

    elif state.classifier == "LinearSVC":
        classifier_params["penalty"] = selectbox_(
            "Specify norm in the penalization:", ["l2", "l1"]
        )
        classifier_params["loss"] = selectbox_(
            "Select loss function:", ["squared_hinge", "hinge"]
        )
        classifier_params["C"] = number_input_(
            "C parameter:", value=1, min_value=1, max_value=100
        )
        classifier_params["cv_generator"] = number_input_(
            "Cross-validation generator:", value=2, min_value=2, max_value=100
        )

    elif state.classifier == "XGBoost":
        classifier_params["learning_rate"] = number_input_(
            "Learning rate:", value=0.3, min_value=0.0, max_value=1.0
        )
        classifier_params["min_split_loss"] = number_input_(
            "Min. split loss:", value=0, min_value=0, max_value=100
        )
        classifier_params["max_depth"] = number_input_(
            "Max. depth:", value=6, min_value=0, max_value=100
        )
        classifier_params["min_child_weight"] = number_input_(
            "Min. child weight:", value=1, min_value=0, max_value=100
        )

    state["classifier_params"] = classifier_params

    # Sidebar -- Cross-Validation
    st.sidebar.markdown(
        "## [Cross-validation](https://omiclearn.readthedocs.io/en/latest//METHODS-%7C-5.-Validation#4-1-cross-validation)"
    )
    state["cv_method"] = selectbox_(
        "Specify CV method:",
        ["RepeatedStratifiedKFold", "StratifiedKFold", "StratifiedShuffleSplit"],
    )
    state["cv_splits"] = number_input_("CV Splits:", min_value=2, max_value=10, value=5)

    # Define placeholder variables for CV
    if state.cv_method == "RepeatedStratifiedKFold":
        state["cv_repeats"] = number_input_(
            "CV Repeats:", min_value=1, max_value=50, value=10
        )

    return state


def session_history(widget_values):
    """
    Helper function to save / show session history
    """

    widget_values['run'] = len(st.session_state.history)+1

    st.session_state.history.append(widget_values)
    sessions_df = pd.DataFrame(st.session_state.history)

    new_column_names = {
        k: v.replace(":", "").replace("Select", "")
        for k, v in zip(sessions_df.columns, sessions_df.columns)
    }
    sessions_df = sessions_df.rename(columns=new_column_names)

    sessions_df = sessions_df.iloc[::-1]

    st.write("## Session History")
    st.dataframe(
    sessions_df.style.format(precision=3)
    )  #  Display only 3 decimal points in UI side
    get_download_link(sessions_df, "session_history.csv")

# Saving session info
def save_sessions(widget_values, user_name):

    session_no, session_dict = get_sessions()
    session_no.append(len(session_no) + 1)
    session_dict[session_no[-1]] = widget_values
    sessions_df = pd.DataFrame(session_dict)
    sessions_df = sessions_df.T
    sessions_df = sessions_df.drop(
        sessions_df[sessions_df["user"] != user_name].index
    ).reset_index(drop=True)
    new_column_names = {
        k: v.replace(":", "").replace("Select", "")
        for k, v in zip(sessions_df.columns, sessions_df.columns)
    }
    sessions_df = sessions_df.rename(columns=new_column_names)
    sessions_df = sessions_df.drop("user", axis=1)

    st.write("## Session History")
    st.dataframe(
        sessions_df.style.format(precision=3)
    )  #  Display only 3 decimal points in UI side
    get_download_link(sessions_df, "session_history.csv")


# Load data
@st.cache(persist=True, show_spinner=True)
def load_data(file_buffer, delimiter):
    """
    Load data to pandas dataframe
    """

    warnings = []
    df = pd.DataFrame()
    if file_buffer is not None:
        if delimiter == "Excel File":
            df = pd.read_excel(file_buffer)

            # check if all columns are strings valid_columns = []
            error = False
            valid_columns = []
            for idx, _ in enumerate(df.columns):
                if isinstance(_, str):
                    valid_columns.append(_)
                else:
                    warnings.append(
                        f"Removing column {idx} with value {_} as type is {type(_)} and not string."
                    )
                    error = True
            if error:
                warnings.append(
                    "Errors detected when importing Excel file. Please check that Excel did not convert protein names to dates."
                )
                df = df[valid_columns]

        elif delimiter == "Comma (,)":
            df = pd.read_csv(file_buffer, sep=",")
        elif delimiter == "Semicolon (;)":
            df = pd.read_csv(file_buffer, sep=";")
        elif delimiter == "Tab (\\t) for TSV":
            df = pd.read_csv(file_buffer, sep="\t")
    return df, warnings


# Show main text and data upload section
def main_text_and_data_upload(state, APP_TITLE):
    st.title(APP_TITLE)

    with st.expander("Disclaimer", expanded=False):

        st.markdown(
            """
        **Note:** It is possible to get artificially high or low performance because of technical and biological artifacts in the data.
        While OmicLearn has the functionality to perform basic exploratory data analysis (EDA) such as PCA, it is not meant to substitute throughout data exploration but rather add a machine learning layer.
        """
        )

        st.markdown(
            """**Note:** By uploading a file, you agree to our
                    [Apache License](https://github.com/MannLabs/OmicLearn/blob/master/LICENSE.txt).
                    Uploaded data will not be saved."""
        )

        citation = "**Reference:** Transparent exploration of machine learning for biomarker discovery from proteomics and omics data\nFurkan M. Torun, Sebastian Virreira Winter, Sophia Doll, Felix M. Riese, Artem Vorobyev, Johannes B. Mueller-Reif, Philipp E. Geyer, Maximilian T. Strauss\nbioRxiv 2021.03.05.434053; doi: https://doi.org/10.1101/2021.03.05.434053"

        st.markdown(citation)


    with st.expander("Upload or select sample dataset (*Required)", expanded=True):

        file_buffer = st.file_uploader(
            "", type=["csv", "xlsx", "xls", "tsv"]
        )

        st.markdown("Maximum size 200 MB. One row per sample, one column per feature. 'Features' (Proteins, Genes, ..) should be uppercase, all additional features with a leading '_'.")

        if file_buffer is not None:
            if file_buffer.name.endswith(".xlsx") or file_buffer.name.endswith(".xls"):
                delimiter = "Excel File"
            elif file_buffer.name.endswith(".tsv"):
                delimiter = "Tab (\\t) for TSV"
            else:
                delimiter = st.selectbox(
                    "Determine the delimiter in your dataset",
                    ["Comma (,)", "Semicolon (;)"],
                )

            df, warnings = load_data(file_buffer, delimiter)

            for warning in warnings:
                st.warning(warning)
            state["df"] = df

        st.markdown("<hr>", unsafe_allow_html=True)

        state["sample_file"] = st.selectbox(
            "Or select sample file here:", ["None", "Alzheimer", "Sample"]
        )

        # Sample dataset / uploaded file selection
        dataframe_length = len(state.df)
        max_df_length = 30

        if state.sample_file != "None" and dataframe_length:
            st.warning(
                "**WARNING:** File uploaded but sample file selected. Please switch sample file to `None` to use your file."
            )
            state["df"] = pd.DataFrame()
        elif state.sample_file != "None":
            if state.sample_file == "Alzheimer":
                st.info(
                    """
                    **This dataset was retrieved from the following paper and the code for parsing is available at
                    [GitHub](https://github.com/MannLabs/OmicLearn/blob/master/data/Alzheimer_paper.ipynb):**\n
                    Bader, J., Geyer, P., Müller, J., Strauss, M., Koch, M., & Leypoldt, F. et al. (2020).
                    Proteome profiling in cerebrospinal fluid reveals novel biomarkers of Alzheimer's disease.
                    Molecular Systems Biology, 16(6). doi: [10.15252/msb.20199356](http://doi.org/10.15252/msb.20199356)
                    """
                )

            folder_to_load = os.path.join(_parent_directory, 'data')
            file_to_load = os.path.join(folder_to_load, state.sample_file + ".xlsx")
            state["df"] = pd.read_excel(file_to_load)
            st.markdown("Using the following dataset:")
            st.dataframe(state.df[state.df.columns[-20:]].head(max_df_length))
        elif 0 < dataframe_length <= max_df_length:
            st.markdown("Using the following dataset:")
            st.dataframe(state.df)
        elif dataframe_length > max_df_length:
            st.markdown("Using the following dataset:")
            st.info(
                f"The dataframe is too large, displaying the first {max_df_length} rows."
            )
            st.dataframe(state.df.head(max_df_length))
        else:
            st.warning("**WARNING:** No dataset uploaded or selected.")

    return state


# Prepare system report
def get_system_report():
    """
    Returns the package versions
    """
    report = {}
    report["omic_learn_version"] = "v1.2"
    report["python_version"] = sys.version[:5]
    report["pandas_version"] = pd.__version__
    report["numpy_version"] = np.version.version
    report["sklearn_version"] = sklearn.__version__
    report["plotly_version"] = plotly.__version__

    return report


# Generate a download link for Plots and CSV
def get_download_link(exported_object, name):
    """
    Generate download link for charts in SVG and PDF formats and for dataframes in CSV format
    """
    os.makedirs("downloads/", exist_ok=True)
    extension = name.split(".")[-1]

    if extension == "svg":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name) as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        href = (
            f'<a class="download_link" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>'
            % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "pdf":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name, "rb") as f:
            pdf = f.read()
        b64 = base64.encodebytes(pdf).decode()
        href = (
            f'<a class="download_link" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>'
            % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "csv":
        exported_object.to_csv("downloads/" + name, index=False)
        with open("downloads/" + name, "rb") as f:
            csv = f.read()
        b64 = base64.b64encode(csv).decode()
        href = (
            f'<a class="download_link" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>'
            % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    else:
        raise NotImplementedError("This output format function is not implemented")


# Generate summary text
def generate_text(state, report):

    text = ""
    # Packages
    packages_plain_text = """
        OmicLearn ({omic_learn_version}) was utilized for performing data analysis, model execution, and creation of plots and charts.
        Machine learning was done in Python ({python_version}). Feature tables were imported via the Pandas package ({pandas_version}) and manipulated using the Numpy package ({numpy_version}).
        The machine learning pipeline was employed using the scikit-learn package ({sklearn_version}).
        The Plotly ({plotly_version}) library was used for plotting.
    """
    text += packages_plain_text.format(**report)

    # Normalization
    if state.normalization == "None":
        text += "No normalization on the data was performed. "
    elif state.normalization in ["StandardScaler", "MinMaxScaler", "RobustScaler"]:
        text += f"Data was normalized in each using a {state.normalization} approach. "
    else:
        params = [f"{k} = {v}" for k, v in state.normalization_params.items()]
        text += f"Data was normalized in each using a {state.normalization} ({' '.join(params)}) approach. "

    # Missing value impt.
    if state.missing_value != "None":
        text += "To impute missing values, a {}-imputation strategy is used. ".format(
            state.missing_value
        )
    else:
        text += "The dataset contained no missing values; hence no imputation was performed. "

    # Features
    if state.feature_method == "None":
        text += "No feature selection algorithm was applied. "
    elif state.feature_method == "ExtraTrees":
        text += "Features were selected using a {} (n_trees={}) strategy with the maximum number of {} features. ".format(
            state.feature_method, state.n_trees, state.max_features
        )
    else:
        text += "Features were selected using a {} strategy with the maximum number of {} features. ".format(
            state.feature_method, state.max_features
        )
    text += "During training, normalization and feature selection was individually performed using the data of each split. "

    # Classification
    params = [f"{k} = {v}" for k, v in state.classifier_params.items()]
    text += f"For classification, we used a {state.classifier}-Classifier ({' '.join(params)}). "

    # Cross-Validation
    if state.cv_method == "RepeatedStratifiedKFold":
        cv_plain_text = """
            When using a repeated (n_repeats={}), stratified cross-validation (RepeatedStratifiedKFold, n_splits={}) approach to classify {} vs. {},
            we achieved a receiver operating characteristic (ROC) with an average AUC (area under the curve) of {:.2f} ({:.2f} std)
            and precision-recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
        """
        text += cv_plain_text.format(
            state.cv_repeats,
            state.cv_splits,
            "".join(state.class_0),
            "".join(state.class_1),
            state.summary.loc["mean"]["roc_auc"],
            state.summary.loc["std"]["roc_auc"],
            state.summary.loc["mean"]["pr_auc"],
            state.summary.loc["std"]["pr_auc"],
        )
    else:
        cv_plain_text = """
            When using a {} cross-validation approach (n_splits={}) to classify {} vs. {}, we achieved a receiver operating characteristic (ROC)
            with an average AUC (area under the curve) of {:.2f} ({:.2f} std) and Precision-Recall (PR) Curve with an average AUC of {:.2f} ({:.2f} std).
        """
        text += cv_plain_text.format(
            state.cv_method,
            state.cv_splits,
            "".join(state.class_0),
            "".join(state.class_1),
            state.summary.loc["mean"]["roc_auc"],
            state.summary.loc["std"]["roc_auc"],
            state.summary.loc["mean"]["pr_auc"],
            state.summary.loc["std"]["pr_auc"],
        )

    if state.cohort_column is not None:
        text += "When training on one cohort and predicting on another to classify {} vs. {}, we achieved the following AUCs: ".format(
            "".join(state.class_0), "".join(state.class_1)
        )
        for i, cohort_combo in enumerate(state.cohort_combos):
            text += "{:.2f} when training on {} and predicting on {} ".format(
                state.cohort_results["roc_auc"][i], cohort_combo[0], cohort_combo[1]
            )
            text += ", and {:.2f} for PR Curve when training on {} and predicting on {}. ".format(
                state.cohort_results["pr_auc"][i], cohort_combo[0], cohort_combo[1]
            )

    # Print the all text
    st.header("Summary")
    with st.expander("Summary text"):
        st.info(text)


# Generate footer
def generate_footer_parts(report):
    st.write('---')
    st.write("[Bug Report (GitHub)](https://github.com/MannLabs/OmicLearn/issues/new/choose)")
