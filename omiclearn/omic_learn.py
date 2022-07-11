"""OmicLearn main file."""
import random
import warnings
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
import os

warnings.simplefilter("ignore")

import omiclearn.utils

# Session State

if 'history' not in st.session_state:
    st.session_state.history = []

# ML functionalities
from omiclearn.utils.ml_helper import perform_cross_validation, transform_dataset, calculate_cm

# Plotting
from omiclearn.utils.plot_helper import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_pr_curve_cv,
    plot_roc_curve_cv,
    perform_EDA,
)

# UI components and others func.
from omiclearn.utils.ui_helper import (
    main_components,
    get_system_report,
    save_sessions,
    load_data,
    main_text_and_data_upload,
    objdict,
    generate_sidebar_elements,
    get_download_link,
    generate_text,
    generate_footer_parts,
    session_history,
)

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)

# Set the configs
APP_TITLE = "OmicLearn — ML platform for omics datasets"
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=Image.open(os.path.join(_this_directory,"utils/omic_learn.ico")),
    layout="centered",
    initial_sidebar_state="auto",
)
icon = Image.open(os.path.join(_this_directory, "utils/omic_learn_black.png"))
report = get_system_report()

# This needs to be here as it needs to be after setting ithe initial_sidebar_state
try:
    import xgboost
except ModuleNotFoundError:
    st.warning(
        "**WARNING:** Xgboost not installed. To use xgboost install using `conda install py-xgboost`"
    )


# Choosing sample dataset and data parameter selections
def checkpoint_for_data_upload(state, record_widgets):
    multiselect = record_widgets.multiselect
    state["n_missing"] = state.df.isnull().sum().sum()

    if len(state.df) > 0:
        if state.n_missing > 0:
            st.info(
                f"**INFO:** Found {state.n_missing} missing values. "
                "Use missing value imputation or `xgboost` classifier."
            )
        # Distinguish the features from others
        state["proteins"] = [_ for _ in state.df.columns.to_list() if _[0] != "_"]
        state["not_proteins"] = [_ for _ in state.df.columns.to_list() if _[0] == "_"]

        # Dataset -- Subset
        with st.expander("Create subset"):
            st.markdown(
                """
                        This section allows you to specify a subset of data based on values within a comma.
                        Hence, you can exclude data that should not be used at all."""
            )
            state["subset_column"] = st.selectbox(
                "Select subset column:", ["None"] + state.not_proteins
            )

            if state.subset_column != "None":
                subset_options = (
                    state.df[state.subset_column].value_counts().index.tolist()
                )
                subset_class = multiselect(
                    "Select values to keep:", subset_options, default=subset_options
                )
                state["df_sub"] = state.df[
                    state.df[state.subset_column].isin(subset_class)
                ].copy()
            elif state.subset_column == "None":
                state["df_sub"] = state.df.copy()
                state["subset_column"] = "None"

        # Dataset -- Feature selections
        with st.expander("Classification target (*Required)"):
            st.markdown(
                """
                Classification target refers to the column that contains the variables that are used two distinguish the two classes.
                In the next section, the unique values of this column can be used to define the two classes.
            """
            )
            state["target_column"] = st.selectbox(
                "Select target column:",
                [""] + state.not_proteins,
                format_func=lambda x: "Select a classification target"
                if x == ""
                else x,
            )
            if state.target_column == "":
                unique_elements_lst = []
            else:
                st.markdown(f"Unique elements in `{state.target_column}` column:")
                unique_elements = state.df_sub[state.target_column].value_counts()
                st.write(unique_elements)
                unique_elements_lst = unique_elements.index.tolist()

        # Dataset -- Class definitions
        with st.expander("Define classes (*Required)"):
            st.markdown(
                f"""
                For a binary classification task, one needs to define two classes based on the
                unique values in the `{state.target_column}` task column.
                It is possible to assign multiple values for each class.
            """
            )
            state["class_0"] = multiselect(
                "Select Class 0:", unique_elements_lst, default=None
            )
            state["class_1"] = multiselect(
                "Select Class 1:",
                [_ for _ in unique_elements_lst if _ not in state.class_0],
                default=None,
            )
            state["remainder"] = [
                _ for _ in state.not_proteins if _ is not state.target_column
            ]

        # Once both classes are defined
        if state.class_0 and state.class_1:

            # EDA Part
            with st.expander("EDA — Exploratory data analysis (^Recommended)"):
                st.markdown(
                    """
                    Use exploratory data anlysis on your dateset to identify potential correlations and biases.
                    For more information, please visit
                    [the dedicated ReadTheDocs page](https://omiclearn.readthedocs.io/en/latest/METHODS.html#exploratory-data-analysis-eda).
                    """
                )
                state["df_sub_y"] = state.df_sub[state.target_column].isin(
                    state.class_0
                )
                state["eda_method"] = st.selectbox(
                    "Select an EDA method:", ["None", "PCA", "Hierarchical clustering"]
                )

                if (state.eda_method == "PCA") and (len(state.proteins) < 6):
                    state["pca_show_features"] = st.checkbox(
                        "Show the feature attributes on the graph", value=False
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

                if (state.eda_method != "None"):
                    with st.spinner(f"Performing {state.eda_method}.."):
                        p = perform_EDA(state)
                        st.plotly_chart(p, use_container_width=True)
                        get_download_link(p, f"{state.eda_method}.pdf")
                        get_download_link(p, f"{state.eda_method}.svg")

            with st.expander("Additional features"):
                st.markdown(
                    "Select additional features. All non numerical values will be encoded (e.g. M/F -> 0,1)"
                )
                state["additional_features"] = multiselect(
                    "Select additional features for trainig:",
                    state.remainder,
                    default=None,
                )

            # Exclude features
            with st.expander("Exclude features"):
                state["exclude_features"] = []
                st.markdown(
                    "Exclude some features from the model training by selecting or uploading a CSV file. "
                    "This can be useful when, e.g., re-running a model without a top feature and assessing the difference in classification accuracy."
                )
                # File uploading target_column for exclusion
                exclusion_file_buffer = st.file_uploader(
                    "Upload your CSV (comma(,) seperated) file here in which each row corresponds to a feature to be excluded.",
                    type=["csv"],
                )
                exclusion_df, exc_df_warnings = load_data(
                    exclusion_file_buffer, "Comma (,)"
                )
                for warning in exc_df_warnings:
                    st.warning(warning)

                if len(exclusion_df) > 0:
                    st.markdown("The following features will be excluded:")
                    st.write(exclusion_df)
                    exclusion_df_list = list(exclusion_df.iloc[:, 0].unique())
                    state["exclude_features"] = multiselect(
                        "Select features to be excluded:",
                        state.proteins,
                        default=exclusion_df_list,
                    )
                else:
                    state["exclude_features"] = multiselect(
                        "Select features to be excluded:", state.proteins, default=[]
                    )

            # Manual feature selection
            with st.expander("Manually select features"):
                st.markdown(
                    "Manually select a subset of features. If only these features should be used, additionally set the "
                    "`Feature selection` method to `None`. Otherwise, feature selection will be applied, and only a subset of the manually selected features is used."
                )
                manual_users_features = multiselect(
                    "Select your features manually:", state.proteins, default=None
                )
            if manual_users_features:
                state.proteins = manual_users_features

        # Dataset -- Cohort selections
        with st.expander("Cohort comparison"):
            st.markdown("Select cohort column to train on one and predict on another:")
            not_proteins_excluded_target_option = state.not_proteins
            if state.target_column != "":
                not_proteins_excluded_target_option.remove(state.target_column)
            state["cohort_column"] = st.selectbox(
                "Select cohort column:", [None] + not_proteins_excluded_target_option
            )
            if state["cohort_column"] == None:
                state["cohort_checkbox"] = None
            else:
                state["cohort_checkbox"] = "Yes"

            if "exclude_features" not in state:
                state["exclude_features"] = []

        state["proteins"] = [
            _ for _ in state.proteins if _ not in state.exclude_features
        ]

    return state


# Display results and plots
def classify_and_plot(state):

    state.bar = st.progress(0)
    # Cross-Validation
    st.markdown("Performing analysis and Running cross-validation")
    cv_results, cv_curves = perform_cross_validation(state)

    st.header("Cross-validation results")

    top_features = []
    # Feature importances from the classifier
    with st.expander("Feature importances from the classifier"):
        st.subheader("Feature importances from the classifier")
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
                st.subheader("Feature importances from classifier table")
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
    # ROC-AUC
    with st.expander(
        "Receiver operating characteristic Curve and Precision-Recall Curve"
    ):
        st.subheader("Receiver operating characteristic")
        p = plot_roc_curve_cv(cv_curves["roc_curves_"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve.pdf")
            get_download_link(p, "roc_curve.svg")

        # Precision-Recall Curve
        st.subheader("Precision-Recall Curve")
        st.markdown(
            "Precision-Recall (PR) Curve might be used for imbalanced datasets."
        )
        p = plot_pr_curve_cv(cv_curves["pr_curves_"], cv_results["class_ratio_test"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "pr_curve.pdf")
            get_download_link(p, "pr_curve.svg")

    # Confusion Matrix (CM)
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
        # (tpr, fpr, tnr, fnr)
        cm_results_ = cm_results.mean().to_frame()
        cm_results_.columns = ["Mean"]

        cm_results_["Std"] = cm_results.std()

        st.write("Average peformance for all splits:")
        st.write(cm_results_)

    # Results table
    with st.expander("Table for run results"):
        st.subheader(f"Run results for `{state.classifier}`")
        state["summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.write(state.summary)
        st.info(
            """
            **Info:** `Mean precision` and `Mean recall` values provided in the table above
            are calculated as the mean of all individual splits shown in the confusion matrix,
            not the "Sum of all splits" matrix.
            """
        )
        get_download_link(state.summary, "run_results.csv")

    if state.cohort_checkbox:
        st.header("Cohort comparison results")
        cohort_results, cohort_curves = perform_cross_validation(
            state, state.cohort_column
        )

        with st.expander(
            "Receiver operating characteristic Curve and Precision-Recall Curve"
        ):
            # ROC-AUC for Cohorts
            st.subheader("Receiver operating characteristic")
            p = plot_roc_curve_cv(
                cohort_curves["roc_curves_"], cohort_curves["cohort_combos"]
            )
            st.plotly_chart(p, use_container_width=True)
            if p:
                get_download_link(p, "roc_curve_cohort.pdf")
                get_download_link(p, "roc_curve_cohort.svg")

            # PR Curve for Cohorts
            st.subheader("Precision-Recall Curve")
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
            st.subheader("Confusion matrix")
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
            st.write(state.cohort_summary)
            get_download_link(state.cohort_summary, "run_results_cohort.csv")

        state["cohort_combos"] = cohort_curves["cohort_combos"]
        state["cohort_results"] = cohort_results

    return state


# Main Function
def OmicLearn_Main():

    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    # Main components
    widget_values, record_widgets = main_components()

    # Welcome text and Data uploading
    state = main_text_and_data_upload(state, APP_TITLE)

    # Checkpoint for whether data uploaded/selected
    state = checkpoint_for_data_upload(state, record_widgets)

    # Sidebar widgets
    state = generate_sidebar_elements(state, icon, report, record_widgets)

    # Analysis Part
    if len(state.df) > 0 and state.target_column == "":
        st.warning("**WARNING:** Select classification target from your data.")

    elif len(state.df) > 0 and not (state.class_0 and state.class_1):
        st.warning("**WARNING:** Define classes for the classification target.")

    elif (
        (state.df is not None)
        and (state.class_0 and state.class_1)
        and (st.button("Run analysis", key="run"))
    ):
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
            f"""
            **Running info:**
            - Using the following features: **Class 0 `{state.class_0}`, Class 1 `{state.class_1}`**.
            - Using classifier **`{state.classifier}`**.
            - Using a total of  **`{len(state.features)}`** features.
            - Note that OmicLearn is intended to be an exploratory tool to assess the performance of algorithms,
                rather than providing a classification model for production.
        """
        )

        # Plotting and Get the results
        state = classify_and_plot(state)

        # Generate summary text
        generate_text(state, report)

        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"

        for _ in state.summary.columns:
            widget_values[_ + "_mean"] = state.summary.loc["mean"][_]
            widget_values[_ + "_std"] = state.summary.loc["std"][_]

        widget_values["top_features"] = state.top_features

        session_history(widget_values)

        # Generate footer
        generate_footer_parts(report)

    else:
        pass


# Run the OmicLearn
if __name__ == "__main__":

    try:
        OmicLearn_Main()
    except (ValueError, IndexError) as val_ind_error:
        st.error(
            f"There is a problem with values/parameters or dataset due to {val_ind_error}."
        )
    except TypeError as e:
        # st.warning("TypeError exists in {}".format(e))
        pass
