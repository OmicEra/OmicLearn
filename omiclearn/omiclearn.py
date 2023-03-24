"""OmicLearn main file."""
import os
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

warnings.simplefilter("ignore")

# Session State
if "history" not in st.session_state:
    st.session_state.history = []

# ML functionalities
from omiclearn.utils.ml_helper import (
    calculate_cm,
    perform_cross_validation,
    transform_dataset,
)

# Plotting
from omiclearn.utils.plot_helper import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_pr_curve_cv,
    plot_roc_curve_cv,
)

# UI components and others func.
from omiclearn.utils.ui_helper import (
    dataset_handling,
    generate_sidebar_elements,
    generate_summary_text,
    get_download_link,
    get_system_report,
    main_text_and_data_upload,
    objdict,
    return_widgets,
    session_history,
)
from omiclearn.utils.ui_texts import (
    APP_TITLE,
    RESULTS_TABLE_INFO,
    RUNNING_INFO_TEXT,
    XGBOOST_NOT_INSTALLED,
)

_this_file = os.path.abspath(__file__)
_this_directory = os.path.dirname(_this_file)

# Set the configs

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=Image.open(os.path.join(_this_directory, "utils/omiclearn.ico")),
    layout="centered",
    initial_sidebar_state="auto",
)
icon = Image.open(os.path.join(_this_directory, "utils/omiclearn_black.png"))
report = get_system_report()

# This needs to be here as it needs to be after setting ithe initial_sidebar_state
try:
    import xgboost
except ModuleNotFoundError:
    st.warning(XGBOOST_NOT_INSTALLED)


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
    # ROC-AUC
    with st.expander("Receiver operating characteristic Curve"):
        p = plot_roc_curve_cv(cv_curves["roc_curves_"])
        st.plotly_chart(p, use_container_width=True)
        if p:
            get_download_link(p, "roc_curve.pdf")
            get_download_link(p, "roc_curve.svg")

    # Precision-Recall Curve
    with st.expander("Precision-Recall Curve"):
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
        cm_results_ = cm_results.mean().to_frame()
        cm_results_.columns = ["Mean"]
        cm_results_["Std"] = cm_results.std()

        st.markdown("**Average peformance for all splits:**")
        st.table(cm_results_)

    # Results table
    with st.expander("Table for run results"):
        st.markdown(f"**Run results for `{state.classifier}` model:**")
        state["summary"] = pd.DataFrame(pd.DataFrame(cv_results).describe())
        st.table(state.summary)
        st.info(RESULTS_TABLE_INFO)
        get_download_link(state.summary, "run_results.csv")

    if state.cohort_checkbox:
        st.header("Cohort comparison results")
        cohort_results, cohort_curves = perform_cross_validation(
            state, state.cohort_column
        )

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

    return state


# Main Function
def OmicLearn_Main():
    # Define state
    state = objdict()
    state["df"] = pd.DataFrame()
    state["class_0"] = None
    state["class_1"] = None

    # Main components
    widget_values, record_widgets = return_widgets()

    # Welcome text and Data uploading
    state = main_text_and_data_upload(state, APP_TITLE)

    # Checkpoint for whether data uploaded/selected
    state = dataset_handling(state, record_widgets)

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
            RUNNING_INFO_TEXT.format(
                STATE_CLASS_0=state.class_0,
                STATE_CLASS_1=state.class_1,
                STATE_CLASSIFIER=state.classifier,
                LEN_STATE_FEATURES=len(state.features),
            )
        )

        # Plotting and Get the results
        state = classify_and_plot(state)

        # Generate summary text
        generate_summary_text(state, report)

        # Session and Run info
        widget_values["Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " (UTC)"

        # Calculate mean, std and get the top features
        for _ in state.summary.columns:
            widget_values[_ + "_mean"] = state.summary.loc["mean"][_]
            widget_values[_ + "_std"] = state.summary.loc["std"][_]
        widget_values["top_features"] = state.top_features

        # Show session history
        session_history(widget_values)

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
