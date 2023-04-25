"""Tests for omiclearn utils."""
import sys
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append("..")
from omiclearn.utils.ml_helper import (
    normalize_dataset,
    transform_dataset,
    perform_cross_validation,
    calculate_cm,
)
from omiclearn.utils.ui_components import load_data, main_analysis_run, objdict
from test_results import *

state = {}


def test_load_data():
    """
    Test the load data function
    Create a file in memory and test loading
    """

    # Excel
    df = pd.DataFrame({"Data": [1, 2, 3, 4]})
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    writer.save()
    xlsx_data, warnings = load_data(output, "Excel File")
    pd.testing.assert_frame_equal(xlsx_data, df)

    # csv
    df = pd.DataFrame({"A": [1, 1], "B": [0, 0]})
    csv_data, warnings = load_data("test_csv_c.csv", "Comma (,)")
    print(csv_data)
    pd.testing.assert_frame_equal(csv_data, df)

    csv_data, warnings = load_data("test_csv_sc.csv", "Semicolon (;)")
    print(csv_data)
    pd.testing.assert_frame_equal(csv_data, df)

    # TSV
    tsv_data, warnings = load_data("test_tsv.tsv", "Tab (\\t) for TSV")
    print(tsv_data)
    pd.testing.assert_frame_equal(tsv_data, df)


def test_transform_dataset():
    """
    Test the transform dataset function
    Test if the transformation is done correctly
    """

    df = pd.DataFrame(
        np.array([[1, 2, "m", "+"], [4, 5, "w", "-"], [7, 8, "m", "-"]]),
        columns=["a", "b", "c", "d"],
    )
    df_t = transform_dataset(df, ["c"], ["a", "b"])
    assert df_t["c"].dtype == np.dtype("int")

    df_t = transform_dataset(df, ["c", "d"], ["a", "b"])
    assert df_t["c"].dtype == np.dtype("int")
    assert df_t["d"].dtype == np.dtype("int")

    df_t = transform_dataset(df, [], ["a", "b"])

    for column in df_t.columns:
        assert df_t[column].dtype == np.dtype("float")


def test_normalize_dataset():
    """
    Tests the normalization
    Calls all the Normalization Methods
    """

    normalization_params = {}
    df = pd.DataFrame({"Data": [1, 2, 3, 4]})

    for normalization in [
        "StandardScaler",
        "MinMaxScaler",
        "RobustScaler",
        "PowerTransformer",
        "QuantileTransformer",
    ]:
        state["normalization"] = normalization
        if normalization == "PowerTransformer":
            normalization_params["method"] = "box-cox"
        elif normalization == "QuantileTransformer":
            del normalization_params["method"]
            normalization_params["random_state"] = 23
            normalization_params["n_quantiles"] = 4
            normalization_params["output_distribution"] = "uniform"
        else:
            pass
        state["normalization_params"] = normalization_params
        normalize_dataset(df, state["normalization"], state["normalization_params"])


def test_integration():
    """Run perform_cross_validation() and compare a previous run.
    - StandardScaler
    - XGBoost with defaults
    - CV: 3 splits * 2 repeats
    - The rest is default.
    - Positive class: a
    - Negative class: b
    - Additional features: _study

    """
    # Define state for Sample.xlsx demo case
    df = pd.read_excel("Sample.xlsx")
    test_state = objdict()
    test_state["df"] = df
    test_state["df_sub"] = df.copy()
    test_state["subset_column"] = "None"
    test_state["target_column"] = "_disease"
    test_state["class_0"] = ["a"]
    test_state["class_1"] = ["b"]
    test_state["df_sub_y"] = df["_disease"].isin(["a"])
    test_state["sample_file"] = "Sample"
    test_state["n_missing"] = "0"
    test_state["proteins"] = ["AAA", "BBB", "CCC"]
    test_state["not_proteins"] = ["_study"]
    test_state["remainder"] = ["_study"]
    test_state["additional_features"] = ["_study"]
    test_state["exclude_features"] = []
    test_state["random_state"] = 23
    test_state["normalization"] = "StandardScaler"
    test_state["normalization_params"] = {}
    test_state["missing_value"] = "None"
    test_state["feature_method"] = "ExtraTrees"
    test_state["max_features"] = 20
    test_state["n_trees"] = 100
    test_state["cohort_column"] = None
    test_state["classifier"] = "XGBoost"
    test_state["classifier_params"] = {
        "random_state": 23,
        "learning_rate": 0.3,
        "min_split_loss": 0,
        "max_depth": 6,
        "min_child_weight": 1,
    }
    test_state["cv_method"] = "RepeatedStratifiedKFold"
    test_state["cv_splits"] = 3
    test_state["cv_repeats"] = 2
    test_state["bar"] = st.progress(0)
    test_state["features"] = ["AAA", "BBB", "CCC", "_study"]
    # Generate X and y
    main_analysis_run(test_state)

    # print("\n", test_state, "\n")
    _cv_results, _cv_curves = perform_cross_validation(test_state, cohort_column=None)
    assert _cv_results == expected_cv_results, "Error in CV Results"
    assert str(_cv_curves) == str(expected_cv_curves_str), "Error in CV Curves"


def test_calculate_cm():
    y_test = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    y_pred = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    expected_tp, expected_fp, expected_tn, expected_fn = 5, 1, 4, 2
    expected_tpr, expected_fpr, expected_tnr, expected_fnr = (
        0.7142857142857143,
        0.2,
        0.8,
        0.2857142857142857,
    )
    (tp, fp, tn, fn), (tpr, fpr, tnr, fnr) = calculate_cm(y_test, y_pred)
    assert (tp, fp, tn, fn) == (
        expected_tp,
        expected_fp,
        expected_tn,
        expected_fn,
    ), "Mistake in CM calculation"
    assert (tpr, fpr, tnr, fnr) == (
        expected_tpr,
        expected_fpr,
        expected_tnr,
        expected_fnr,
    ), "Mistake in CM rate calculation"
