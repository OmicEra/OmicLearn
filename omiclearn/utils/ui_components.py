"""OmicLearn UI components."""
import base64
import os
import sys

import numpy as np
import pandas as pd
import plotly
import sklearn
import streamlit as st

from .ml_helper import calculate_cm, perform_cross_validation, transform_dataset
from .plot_helper import (
    perform_EDA,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_pr_curve_cv,
    plot_roc_curve_cv,
)
from .ui_texts import *

# Checkpoint for XGBoost
xgboost_installed = False
try:
    import xgboost
    from xgboost import XGBClassifier

    xgboost_installed = True
except ModuleNotFoundError:
    pass

# Define paths
_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)
_parent_directory = os.path.dirname(_this_directory)


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


# Object for state dict
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
def return_widgets():
    """
    Create and return widgets
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


# Generate normalization elements for sidebar
def _generate_normalization_elements(state, selectbox_, number_input_):
    # Preprocessing -- Normalization
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

    # Normalization -- Paremeters selection
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
    # Save the normalization params
    state["normalization_params"] = normalization_params


# Generate missing value imputation elements for sidebar
def _generate_imputation_elements(state, selectbox_):
    # Preprocessing -- Missing value imputation
    if state.n_missing > 0:
        st.sidebar.markdown(
            "## [Missing value imputation](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#imputation-of-missing-values)"
        )
        missing_values = ["Zero", "Mean", "Median", "KNNImputer", "None"]
        state["missing_value"] = selectbox_("Missing value imputation", missing_values)
    else:
        state["missing_value"] = "None"


# Generate feature selection elements for sidebar
def _generate_feature_selection_elements(state, selectbox_, number_input_):
    st.sidebar.markdown(
        "## [Feature selection](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#feature-selection)"
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
            "Number of trees in the forest:",
            value=100,
            min_value=1,
            max_value=2000,
        )
    else:
        state["n_trees"] = 0


# Generate classification method selection elements for sidebar
def _generate_classification_elements(
    state,
    selectbox_,
    number_input_,
):
    st.sidebar.markdown(
        "## [Classification](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#classification)"
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

    # Classification method -- Hyperparameter selection
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
            "Specify norm in the penalization:",
            ["l2", "l1", "ElasticNet", "None"],
        ).lower()
        classifier_params["solver"] = selectbox_(
            "Select the algorithm for optimization:",
            ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
        )
        classifier_params["max_iter"] = number_input_(
            "Maximum number of iteration:",
            value=100,
            min_value=1,
            max_value=2000,
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

    # Save the classification hyperparameters
    state["classifier_params"] = classifier_params


# Generate cross-validation method selection elements for sidebar
def _generate_cross_validation_elements(state, selectbox_, number_input_):
    st.sidebar.markdown(
        "## [Cross-validation](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#validation)"
    )
    state["cv_method"] = selectbox_(
        "Specify CV method:",
        [
            "RepeatedStratifiedKFold",
            "StratifiedKFold",
            "StratifiedShuffleSplit",
        ],
    )
    state["cv_splits"] = number_input_("CV Splits:", min_value=2, max_value=10, value=5)

    # Define placeholder variables for CV
    if state.cv_method == "RepeatedStratifiedKFold":
        state["cv_repeats"] = number_input_(
            "CV Repeats:", min_value=1, max_value=50, value=10
        )


# Generate sidebar elements
def generate_sidebar_elements(state, icon, report, record_widgets):
    slider_ = record_widgets.slider_
    selectbox_ = record_widgets.selectbox_
    number_input_ = record_widgets.number_input_

    # Image/Title
    st.sidebar.image(
        icon,
        use_column_width=True,
        caption="OmicLearn " + report["OmicLearn_version"],
    )
    st.sidebar.markdown(
        "# [Options](https://OmicLearn.readthedocs.io/en/latest//METHODS.html)"
    )

    # Random State
    state["random_state"] = slider_(
        "Random State:", min_value=0, max_value=99, value=23
    )

    # Preprocessing (Normalization and Missing Value Imputation)
    st.sidebar.markdown(
        "## [Preprocessing](https://OmicLearn.readthedocs.io/en/latest/METHODS.html#preprocessing)"
    )
    _generate_normalization_elements(state, selectbox_, number_input_)
    _generate_imputation_elements(state, selectbox_)

    # Feature Selection
    _generate_feature_selection_elements(state, selectbox_, number_input_)

    # Classification Method Selection
    _generate_classification_elements(state, selectbox_, number_input_)

    # Cross-Validation
    _generate_cross_validation_elements(state, selectbox_, number_input_)

    return state


# Get session history
def session_history(widget_values):
    """
    Helper function to save / show session history
    """

    widget_values["run"] = len(st.session_state.history) + 1
    st.session_state.history.append(widget_values)
    sessions_df = pd.DataFrame(st.session_state.history)
    new_column_names = {
        k: v.replace(":", "").replace("Select", "")
        for k, v in zip(sessions_df.columns, sessions_df.columns)
    }
    sessions_df = sessions_df.rename(columns=new_column_names)
    sessions_df = sessions_df.iloc[::-1]

    st.header("Session History")
    st.dataframe(sessions_df.style.format(precision=3))
    get_download_link(sessions_df, "session_history.csv")


# Load data
@st.cache_data(persist=True, show_spinner=True)
def load_data(file_buffer, delimiter, header="infer"):
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
            df = pd.read_csv(file_buffer, sep=",", header=header)
        elif delimiter == "Semicolon (;)":
            df = pd.read_csv(file_buffer, sep=";")
        elif delimiter == "Tab (\\t) for TSV":
            df = pd.read_csv(file_buffer, sep="\t")
    return df, warnings


# Generate tab elements for disclaimer, citation and bug report
def _generate_tabs_elements():
    with st.expander(f"Disclaimer and Citation"):
        disc_tab, citation_tab, bug_tab = st.tabs(
            ["Disclaimer", "Citation", "Bug report"]
        )
        with disc_tab:
            st.markdown(DISCLAIMER_TEXT)
        with citation_tab:
            st.markdown(CITATION_TEXT)
        with bug_tab:
            st.markdown(BUG_REPORT_TEXT)


# Show main text and data upload section
def main_text_and_data_upload(state, APP_TITLE):
    # App title
    st.title(APP_TITLE)

    # Tabs
    _generate_tabs_elements()

    # File upload or file selection
    with st.expander("Upload or select sample dataset (*Required)", expanded=True):
        file_buffer = st.file_uploader("", type=["csv", "xlsx", "xls", "tsv"])
        st.markdown(FILE_UPLOAD_TEXT)

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
                st.info(ALZHEIMER_CITATION_TEXT)

            folder_to_load = os.path.join(_parent_directory, "data")
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


# Generate data subset section
def _generate_subset_section(state, multiselect):
    with st.expander("Create subset"):
        st.markdown(SUBSET_TEXT)
        state["subset_column"] = st.selectbox(
            "Select subset column:",
            ["None"] + state.not_proteins,
        )

        if state.subset_column != "None":
            subset_options = state.df[state.subset_column].value_counts().index.tolist()
            subset_class = multiselect(
                "Select values to keep:",
                subset_options,
                default=subset_options,
            )
            state["df_sub"] = state.df[
                state.df[state.subset_column].isin(subset_class)
            ].copy()
        elif state.subset_column == "None":
            state["df_sub"] = state.df.copy()
            state["subset_column"] = "None"


# Generate classification target selection section
def _generate_classification_target_section(state):
    with st.expander("Classification target (*Required)"):
        st.markdown(CLASSIFICATION_TARGET_TEXT)
        state["target_column"] = st.selectbox(
            "Select target column:",
            [""] + state.not_proteins,
            format_func=lambda x: "Select a classification target" if x == "" else x,
        )
        if state.target_column == "":
            unique_elements_list = []
        else:
            st.markdown(f"Unique elements in **`{state.target_column}`** column:")
            unique_elements = state.df_sub[state.target_column].value_counts()
            st.table(unique_elements)
            unique_elements_list = unique_elements.index.tolist()
        return unique_elements_list


# Generate classification classes selection section
def _generate_class_selections(state, multiselect, unique_elements_list):
    with st.expander("Define classes (*Required)"):
        st.markdown(DEFINE_CLASS_TEXT.format(STATE_TARGET_COLUMN=state.target_column))
        state["class_0"] = multiselect(
            "Select Positive Class:",
            unique_elements_list,
            default=None,
            help="Select the experiment group like cancer, diseased or drug-applied group as positive class.",
        )
        state["class_1"] = multiselect(
            "Select Negative Class:",
            [_ for _ in unique_elements_list if _ not in state.class_0],
            default=None,
            help="Select the control/healthy group as negative class.",
        )
        state["remainder"] = [
            _ for _ in state.not_proteins if _ is not state.target_column
        ]


# Generate exploratory data analysis section
def _generate_eda_section(state):
    with st.expander("EDA — Exploratory data analysis (^Recommended)"):
        st.markdown(EDA_TEXT)
        state["df_sub_y"] = state.df_sub[state.target_column].isin(state.class_0)
        state["eda_method"] = st.selectbox(
            "Select an EDA method:",
            ["None", "PCA", "Hierarchical clustering"],
        )

        if (state.eda_method == "PCA") and (len(state.proteins) < 6):
            state["pca_show_features"] = st.checkbox(
                "Show the feature attributes on the graph",
                value=False,
            )

        if state.eda_method == "Hierarchical clustering":
            state["data_range"] = st.slider(
                "Data range to be visualized",
                0,
                len(state.proteins),
                (0, round(len(state.proteins) / 2)),
                step=3,
                help="In large datasets, it is not possible to visaulize all the features.",
            )

        if state.eda_method != "None":
            with st.spinner(f"Performing {state.eda_method}.."):
                p = perform_EDA(state)
                st.plotly_chart(p, use_container_width=True)
                get_download_link(p, f"{state.eda_method}.pdf")
                get_download_link(p, f"{state.eda_method}.svg")


# Generate additional feature selection section
def _generate_additional_feature_selection_section(state, multiselect):
    with st.expander("Additional features"):
        st.markdown(ADDITIONAL_FEATURES_TEXT)

        # File uploading for features to be excluded
        additional_features_file_buffer = st.file_uploader(
            "Upload your CSV (comma(,) seperated) file here in which each row corresponds to an additional feature to be included for training.",
            help="Upload your CSV (comma(,) seperated) file here in which each row corresponds to an additional feature to be included for training.",
            type=["csv"],
        )
        additional_features_df, add_df_warnings = load_data(
            additional_features_file_buffer, "Comma (,)", header=None
        )
        for warning in add_df_warnings:
            st.warning(warning)

        if len(additional_features_df) > 0:
            st.markdown(
                "The following additional features will be included for training:"
            )
            additional_features_df.columns = [
                "Additional features to be included for training"
            ]
            st.table(additional_features_df)
            additional_features_df_list = list(
                additional_features_df.iloc[:, 0].unique()
            )
            suitable_uploaded_features = [
                _ for _ in additional_features_df_list if _ in state.remainder
            ]
            if len(suitable_uploaded_features) != len(additional_features_df_list):
                st.warning(FEATURES_UPLOAD_WARNING_TEXT)

            # Selectbox with uploaded available features
            state["additional_features"] = multiselect(
                "Select additional features for training:",
                help="Select additional features for training:",
                options=state.remainder,
                default=suitable_uploaded_features,
            )
        else:
            state["additional_features"] = multiselect(
                "Select additional features for training:",
                help="Select additional features to be included for training:",
                options=state.remainder,
                default=None,
            )


# Generate exclude features selection section
def _generate_exclude_features_section(state, multiselect):
    with st.expander("Exclude features"):
        state["exclude_features"] = []
        st.markdown(EXCLUDE_FEATURES_TEXT)
        # File uploading for features to be excluded
        exclusion_file_buffer = st.file_uploader(
            "Upload your CSV (comma(,) seperated) file here in which each row corresponds to a feature to be excluded.",
            help="Upload your CSV (comma(,) seperated) file here in which each row corresponds to a feature to be excluded.",
            type=["csv"],
        )
        exclusion_df, exc_df_warnings = load_data(
            exclusion_file_buffer, "Comma (,)", header=None
        )
        for warning in exc_df_warnings:
            st.warning(warning)

        if len(exclusion_df) > 0:
            st.markdown("The following features will be excluded:")
            exclusion_df.columns = ["Features to be excluded from training"]
            st.table(exclusion_df)
            exclusion_df_list = list(exclusion_df.iloc[:, 0].unique())

            suitable_uploaded_features = [
                _ for _ in exclusion_df_list if _ in state.proteins
            ]
            if len(suitable_uploaded_features) != len(exclusion_df_list):
                st.warning(FEATURES_UPLOAD_WARNING_TEXT)

            state["exclude_features"] = multiselect(
                "Select features to be excluded:",
                help="Select features to be excluded from training:",
                options=state.proteins,
                default=suitable_uploaded_features,
            )
        else:
            state["exclude_features"] = multiselect(
                "Select features to be excluded:",
                help="Select features to be excluded from training:",
                options=state.proteins,
                default=[],
            )


# Generate manual feature selection section
def _generate_manual_feature_selection_section(state, multiselect):
    with st.expander("Manually select features"):
        st.markdown(MANUALLY_SELECT_FEATURES_TEXT)
        manual_users_features = multiselect(
            "Select your features manually:",
            state.proteins,
            default=None,
        )
    if manual_users_features:
        state.proteins = manual_users_features


# Generate cohort comparison section
def _generate_cohort_comparison_section(state):
    with st.expander("Cohort comparison"):
        st.markdown("Select cohort column to train on one and predict on another:")
        not_proteins_excluded_target_option = state.not_proteins
        if state.target_column != "":
            not_proteins_excluded_target_option.remove(state.target_column)
        state["cohort_column"] = st.selectbox(
            "Select cohort column:",
            [None] + not_proteins_excluded_target_option,
        )
        if state["cohort_column"] == None:
            state["cohort_checkbox"] = None
        else:
            state["cohort_checkbox"] = "Yes"


# Dataset handling all parts
def dataset_handling(state, record_widgets):
    multiselect = record_widgets.multiselect
    state["n_missing"] = state.df.isnull().sum().sum()

    if len(state.df) > 0:
        if state.n_missing > 0:
            st.info(
                f"**INFO:** Found {state.n_missing} missing values. "
                "Use missing value imputation or **XGBoost** classifier."
            )
        # Distinguish the features from others
        state["proteins"] = [_ for _ in state.df.columns.to_list() if _[0] != "_"]
        state["not_proteins"] = [_ for _ in state.df.columns.to_list() if _[0] == "_"]

        # Create subset section
        _generate_subset_section(state, multiselect)

        # Classification target selection section
        unique_elements_list = _generate_classification_target_section(state)

        # Class definitions section
        _generate_class_selections(state, multiselect, unique_elements_list)

        # Once both classes are defined
        if state.class_0 and state.class_1:
            # EDA section
            _generate_eda_section(state)

            # Additional features selection section
            _generate_additional_feature_selection_section(state, multiselect)

            # Exclude features section
            _generate_exclude_features_section(state, multiselect)

            # Manual feature selection section
            _generate_manual_feature_selection_section(state, multiselect)

            # Cohort comparison section
            _generate_cohort_comparison_section(state)

        # Define excluded features and proteins list
        if "exclude_features" not in state:
            state["exclude_features"] = []
        state["proteins"] = [
            _ for _ in state.proteins if _ not in state.exclude_features
        ]

    return state


# Main analysis run section
def main_analysis_run(state):
    state.features = state.proteins + state.additional_features
    subset = state.df_sub[
        state.df_sub[state.target_column].isin(state.class_0)
        | state.df_sub[state.target_column].isin(state.class_1)
    ].copy()
    state.y = subset[state.target_column].isin(state.class_0)
    state.X = transform_dataset(subset, state.additional_features, state.proteins)

    if state.cohort_column is not None:
        state["X_cohort"] = subset[state.cohort_column]

    # Show the running info text
    st.info(
        RUNNING_INFO_TEXT.format(
            STATE_CLASS_0=state.class_0,
            STATE_CLASS_1=state.class_1,
            STATE_CLASSIFIER=state.classifier,
            LEN_STATE_FEATURES=len(state.features),
        )
    )


# Prepare system report
def get_system_report():
    """
    Returns the package versions
    """
    report = {}
    report["OmicLearn_version"] = "v1.4"
    report["python_version"] = sys.version[:5]
    report["pandas_version"] = pd.__version__
    report["numpy_version"] = np.version.version
    report["sklearn_version"] = sklearn.__version__
    report["plotly_version"] = plotly.__version__
    report["xgboost_version"] = xgboost.__version__
    return report


# Get download link for plots, CSV and TXT
def get_download_link(exported_object, name):
    """
    Generate download link for charts in SVG and PDF formats and for dataframes in CSV format
    """
    os.makedirs("downloads/", exist_ok=True)
    extension = name.split(".")[-1]
    download_button_css_class = "css-1x8cf1d edgvbvh10"

    if extension == "svg":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name) as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode()).decode()
        href = (
            f'<a class="{download_button_css_class}" href="data:image/svg+xml;base64,%s" download="%s" >Download as *.svg</a>'
            % (b64, name)
        )
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "pdf":
        exported_object.write_image("downloads/" + name, height=700, width=700, scale=1)
        with open("downloads/" + name, "rb") as f:
            pdf = f.read()
        b64 = base64.encodebytes(pdf).decode()
        href = (
            f'<a class="{download_button_css_class}" href="data:application/pdf;base64,%s" download="%s" >Download as *.pdf</a>'
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
            f'<a class="{download_button_css_class}" href="data:file/csv;base64,%s" download="%s" >Download as *.csv</a>'
            % (b64, name)
        )
        st.markdown("")
        st.markdown(href, unsafe_allow_html=True)

    elif extension == "txt":
        with open("downloads/" + name, "w") as f:
            f.write(exported_object.replace("  ", ""))
        with open("downloads/" + name, "rb") as f:
            txt = f.read()
        b64 = base64.b64encode(txt).decode()
        href = (
            f'<a class="{download_button_css_class}" href="data:text/plain;base64,%s" download="%s" >Download as *.txt</a>'
            % (b64, name)
        )
        st.markdown(href, unsafe_allow_html=True)

    else:
        raise NotImplementedError("This output format function is not implemented")


# Generate summary text
def generate_summary_text(state, report):
    text = ""
    # Packages
    text += PACKAGES_PLAIN_TEXT.format(**report)

    # Normalization
    if state.normalization == "None":
        text += "No normalization on the data was performed. "
    elif state.normalization in [
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
    ]:
        text += f"Data was normalized in each using a {state.normalization} approach. "
    else:
        params = [f"{k} = {v}" for k, v in state.normalization_params.items()]
        text += f"Data was normalized in each using a {state.normalization} ({', '.join(params)}) approach. "

    # Missing value imptutation
    if state.n_missing > 0:
        if state.missing_value != "None":
            text += f"To impute missing values, a {state.missing_value}-imputation strategy is used. "
        else:
            text += "Even though dataset contained missing values; no imputation was performed. "
    else:
        text += "The dataset contained no missing values; hence no imputation was performed. "

    # Features
    if state.feature_method == "None":
        text += "No feature selection algorithm was applied. "
    elif state.feature_method == "ExtraTrees":
        text += f"Features were selected using a {state.feature_method} (n_trees={state.n_trees}) strategy with the maximum number of {state.max_features} features. "
    else:
        text += f"Features were selected using a {state.feature_method} strategy with the maximum number of {state.max_features} features. "
    text += "During training, normalization and feature selection was individually performed using the data of each split. "

    # Classification
    params = [f"{k} = {v}" for k, v in state.classifier_params.items()]
    if state.classifier == "XGBoost":
        text += f"For classification, we used a {state.classifier}-Classifier (Version: {report['xgboost_version']}, {' '.join(params)}). "
    else:
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
                state.cohort_results["roc_auc"][i],
                cohort_combo[0],
                cohort_combo[1],
            )
            text += ", and {:.2f} for PR Curve when training on {} and predicting on {}. ".format(
                state.cohort_results["pr_auc"][i],
                cohort_combo[0],
                cohort_combo[1],
            )

    # Print the all text
    st.header("Summary")
    with st.expander("Summary text"):
        st.info(text)
        get_download_link(text, "summary_text.txt")


# Display feature importances
def _generate_feature_importances_section(state, cv_curves):
    top_features = []
    # Feature importances from the classifier
    with st.expander("Feature importances from the classifier"):
        if state.cv_method == "RepeatedStratifiedKFold":
            st.markdown(
                f"This is the average feature importance from all {state.cv_splits*state.cv_repeats} cross validation runs."
            )
        else:
            st.markdown(
                f"This is the average feature importance from all {state.cv_splits} cross validation runs."
            )

        if cv_curves["feature_importances_"] is not None:
            # Check whether all feature importance attributes are 0 or not
            if (
                pd.DataFrame(cv_curves["feature_importances_"]).isin([0]).all().all()
                == False
            ):
                p, feature_df, feature_df_wo_links = plot_feature_importance(
                    cv_curves["feature_importances_"]
                )
                st.plotly_chart(p, use_container_width=True)
                if p:
                    get_download_link(p, "clf_feature_importance.pdf")
                    get_download_link(p, "clf_feature_importance.svg")

                # Display `feature_df` with NCBI links
                st.write(
                    feature_df.to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
                get_download_link(feature_df_wo_links, "clf_feature_importances.csv")

                top_features = feature_df.index.to_list()

            else:
                st.info(
                    "All feature importance attribute are zero (0). The plot and table are not displayed."
                )
        else:
            st.info(
                "Feature importance attribute is not implemented for this classifier."
            )
    state["top_features"] = top_features


# Generate ROC section
def _generate_roc_curve_section(cv_curves):
    with st.expander("Receiver operating characteristic Curve"):
        p = plot_roc_curve_cv(cv_curves["roc_curves_"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve.pdf")
            get_download_link(p, "roc_curve.svg")


# Generate PR curve
def _generate_pr_curve_section(cv_curves, cv_results):
    with st.expander("Precision-Recall Curve"):
        st.markdown(
            "Precision-Recall (PR) Curve might be used for imbalanced datasets."
        )
        p = plot_pr_curve_cv(cv_curves["pr_curves_"], cv_results["class_ratio_test"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "pr_curve.pdf")
            get_download_link(p, "pr_curve.svg")


# Generate confusion matrix
def _generate_cm_section(state, cv_curves):
    with st.expander("Confusion matrix"):
        names = ["CV_split {}".format(_ + 1) for _ in range(len(cv_curves["y_hats_"]))]
        names.insert(0, "Sum of all splits")
        p = plot_confusion_matrices(
            state.class_0, state.class_1, cv_curves["y_hats_"], names
        )
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "cm.pdf")
            get_download_link(p, "cm.svg")

        cm_results = [calculate_cm(*_)[1] for _ in cv_curves["y_hats_"]]
        cm_results = pd.DataFrame(cm_results, columns=["TPR", "FPR", "TNR", "FNR"])
        cm_results_ = cm_results.mean().to_frame()
        cm_results_.columns = ["Mean"]
        cm_results_["Std"] = cm_results.std()

        st.markdown("**Average peformance for all splits:**")
        st.table(cm_results_)


# Display results table
def _generate_results_table_section(state, cv_results):
    with st.expander("Table for run results"):
        st.markdown(f"**Run results for `{state.classifier}` model:**")
        state["summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.table(state.summary)
        st.info(RESULTS_TABLE_INFO)
        get_download_link(state.summary, "run_results.csv")


# Display cohort results
def _generate_cohort_results_section(state, cv_results):
    st.header("Cohort comparison results")
    cohort_results, cohort_curves = perform_cross_validation(state, state.cohort_column)

    # ROC-AUC for Cohorts
    with st.expander("Receiver operating characteristic Curve"):
        p = plot_roc_curve_cv(
            cohort_curves["roc_curves_"], cohort_curves["cohort_combos"]
        )
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve_cohort.pdf")
            get_download_link(p, "roc_curve_cohort.svg")

    # PR Curve for Cohorts
    with st.expander("Precision-Recall Curve"):
        st.markdown(
            "Precision-Recall (PR) Curve might be used for imbalanced datasets."
        )
        p = plot_pr_curve_cv(
            cohort_curves["pr_curves_"],
            cohort_results["class_ratio_test"],
            cohort_curves["cohort_combos"],
        )
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "pr_curve_cohort.pdf")
            get_download_link(p, "pr_curve_cohort.svg")

    # Confusion Matrix (CM) for Cohorts
    with st.expander("Confusion matrix"):
        names = [
            "Train on {}, Test on {}".format(_[0], _[1])
            for _ in cohort_curves["cohort_combos"]
        ]
        names.insert(0, "Sum of cohort comparisons")

        p = plot_confusion_matrices(
            state.class_0, state.class_1, cohort_curves["y_hats_"], names
        )
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "cm_cohorts.pdf")
            get_download_link(p, "cm_cohorts.svg")

    with st.expander("Table for run results"):
        state["cohort_summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.table(state.cohort_summary)
        get_download_link(state.cohort_summary, "run_results_cohort.csv")

    state["cohort_combos"] = cohort_curves["cohort_combos"]
    state["cohort_results"] = cohort_results


# Display all results and plots
def display_results_and_plots(state):
    state.bar = st.progress(0)

    # Cross-Validation
    st.markdown("Performing analysis and Running cross-validation")
    cv_results, cv_curves = perform_cross_validation(state)
    st.header("Cross-validation results")

    # Feature importances
    _generate_feature_importances_section(state, cv_curves)

    # ROC-AUC
    _generate_roc_curve_section(cv_curves)

    # Precision-Recall Curve
    _generate_pr_curve_section(cv_curves, cv_results)

    # Confusion Matrix (CM)
    _generate_cm_section(state, cv_curves)

    # Results table
    _generate_results_table_section(state, cv_results)

    # Cohort results
    if state.cohort_checkbox:
        _generate_cohort_results_section(state, cv_results)

    return state
