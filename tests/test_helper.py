"""Tests for OmicLearn utils."""
import sys
from io import BytesIO

import numpy as np
import pandas as pd

sys.path.append("..")
from omiclearn.utils.ml_helper import normalize_dataset, transform_dataset
from omiclearn.utils.ui_helper import load_data

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
