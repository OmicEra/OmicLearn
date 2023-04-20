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

# UI components and others func.
from utils.ui_components import (
    dataset_handling,
    display_results_and_plots,
    generate_sidebar_elements,
    generate_summary_text,
    get_system_report,
    main_analysis_run,
    main_text_and_data_upload,
    objdict,
    return_widgets,
    session_history,
)

from utils.ui_texts import *

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
        # Run main analysis
        main_analysis_run(state)

        # Display all results and plots
        state = display_results_and_plots(state)

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
