# Plotly Graphs
from itertools import chain

# Others
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import auc

# ML funcs
from .ml_helper import calculate_cm

# Define common colors
blue_color = "#035672"
red_color = "#f84f57"
gray_color = "#ccc"


# Prepare feature importance chart
def plot_feature_importance(feature_importance):
    """
    Creates a Plotly barplot to plot feature importance
    """
    fi = [pd.DataFrame.from_dict(_, orient="index") for _ in feature_importance]
    feature_df_ = pd.concat(fi)
    feature_df = feature_df_.groupby(feature_df_.index).sum()
    feature_df_std = feature_df_.groupby(feature_df_.index).std()
    feature_df_std = feature_df_std / feature_df_std.sum() / feature_df.sum()
    feature_df.columns = ["Feature_importance"]
    feature_df = feature_df / feature_df.sum()
    feature_df["Std"] = feature_df_std.values

    feature_df = feature_df.sort_values(by="Feature_importance", ascending=False)
    feature_df = feature_df[feature_df["Feature_importance"] > 0]
    feature_df["Name"] = feature_df.index

    display_limit = 20
    if len(feature_df) > display_limit:
        remainder = pd.DataFrame(
            {
                "Feature_importance": [feature_df.iloc[display_limit:].sum().values[0]],
                "Name": "Remainder",
            },
            index=["Remainder"],
        )
        feature_df = feature_df.iloc[
            :display_limit
        ]  # Show at most `display_limit` entries
        feature_df = feature_df.append(remainder)

    feature_df["Feature_importance"] = (
        feature_df["Feature_importance"].map("{:.3f}".format).astype(np.float32)
    )
    feature_df["Std"] = feature_df["Std"].map("{:.5f}".format)
    feature_df_wo_links = feature_df.copy()
    feature_df["Name"] = feature_df["Name"].apply(
        lambda x: '<a href="https://www.ncbi.nlm.nih.gov/search/all/?term={}" title="Search on NCBI" target="_blank">{}</a>'.format(
            x, x
        )
        if not x.startswith("_") and x != "Remainder"
        else x
    )
    feature_df["Plot_Name"] = feature_df_wo_links["Name"].apply(
        lambda x: '<a href="https://www.ncbi.nlm.nih.gov/search/all/?term={}" title="Search on NCBI" target="_blank">{}</a>'.format(
            x, x if len(x) < 20 else x[:20] + ".."
        )
        if not x.startswith("_") and x != "Remainder"
        else x
    )
    marker_color = red_color
    title = "Top features from the classifier"
    labels = {
        "Feature_importance": "Feature importances from the classifier",
        "Plot_Name": "Names",
        "Std": "Standard Deviation",
    }

    # Hide pvalue if it does not exist
    hover_data = {
        "Plot_Name": False,
        "Name": True,
        "Feature_importance": True,
        "Std": True,
    }
    p = px.bar(
        feature_df.iloc[::-1],
        x="Feature_importance",
        y="Plot_Name",
        orientation="h",
        hover_data=hover_data,
        labels=labels,
        height=800,
        title=title,
    )
    p.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        showlegend=False,
    )
    p.update_traces(marker_color=marker_color)
    p.update_xaxes(showline=True, linewidth=1, linecolor="black")
    p.update_yaxes(showline=True, linewidth=1, linecolor="black", type="category")

    # Update `feature_df` for NaN values and column naming, ordering
    feature_df.dropna(axis="columns", how="all", inplace=True)
    feature_df.drop("Plot_Name", inplace=True, axis=1)
    feature_df_wo_links.dropna(axis="columns", how="all", inplace=True)
    feature_df.rename(
        columns={
            "Name": "Name and NCBI Link",
            "Feature_importance": "Feature Importance",
            "Std": "Standard Deviation",
        },
        inplace=True,
    )

    return (
        p,
        feature_df[["Name and NCBI Link", "Feature Importance", "Standard Deviation"]],
        feature_df_wo_links,
    )


# Prepare confusion matrix plot
def plot_confusion_matrices(class_0, class_1, results, names):
    "Returns Plotly chart for confusion matrices"
    cm_results = [calculate_cm(*_) for _ in results]
    # also include a summary confusion_matrix
    y_test_ = np.array(list(chain.from_iterable([_[0] for _ in results])))
    y_pred_ = np.array(list(chain.from_iterable([_[1] for _ in results])))

    cm_results.insert(0, calculate_cm(y_test_, y_pred_))
    texts = []
    for j in cm_results:
        texts.append(["{}\n{:.0f} %".format(_[0], _[1] * 100) for _ in zip(*j)])
    cats = ["_".join(class_0), "_".join(class_1)]

    x_ = [cats[0], cats[0], cats[1], cats[1]]
    y_ = [cats[0], cats[1], cats[1], cats[0]]

    #  Heatmap
    custom_colorscale = [[0, "#e8f1f7"], [1, "#3886bc"]]
    data = [
        go.Heatmap(
            x=x_,
            y=y_,
            z=cm_results[step][1],
            visible=False,
            hoverinfo="none",
            colorscale=custom_colorscale,
        )
        for step in range(len(cm_results))
    ]
    data[0]["visible"] = True

    # Build slider steps
    steps = []
    for i in range(len(data)):
        step = dict(
            method="update",
            args=[
                # Make the i'th trace visible
                {"visible": [t == i for t in range(len(data))]},
                {
                    "annotations": [
                        dict(
                            x=x_[k],
                            y=y_[k],
                            xref="x1",
                            yref="y1",
                            showarrow=False,
                            text=texts[i][k].replace("\n", "<br>"),
                            font=dict(size=16, color="black"),
                        )
                        for k in range(len(x_))
                    ]
                },
            ],
            label=names[i],
        )
        steps.append(step)

    layout_plotly = {
        "xaxis": {"title": "Predicted value"},
        "yaxis": {"title": "True value"},
        "annotations": steps[0]["args"][1]["annotations"],
        "plot_bgcolor": "rgba(0,0,0,0)",
    }
    p = go.Figure(data=data, layout=layout_plotly)

    # Add slider
    sliders = [
        dict(
            currentvalue={"prefix": "CV Split: "},
            pad={"t": 72},
            active=0,
            steps=steps,
        )
    ]
    p.layout.update(sliders=sliders)
    p.update_layout(autosize=False, width=700, height=700)

    return p


# Prepare ROC Curve
def plot_roc_curve_cv(roc_curve_results, cohort_combos=None):
    """
    Plotly chart for roc curve for cross validation
    """
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    roc_aucs = []
    p = go.Figure()

    for idx, (fpr, tpr, threshold) in enumerate(roc_curve_results):
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        if cohort_combos is not None:
            text = "Train: {} <br>Test: {}".format(
                cohort_combos[idx][0], cohort_combos[idx][1]
            )
            hovertemplate = (
                "False positive rate: %{x:.2f} <br>True positive rate: %{y:.2f}"
                + "<br>"
                + text
            )
            p.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    hovertemplate=hovertemplate,
                    hoverinfo="all",
                    mode="lines",
                    name="Train on {}, Test on {}, AUC {:.2f}".format(
                        cohort_combos[idx][0], cohort_combos[idx][1], roc_auc
                    ),
                )
            )
        else:
            pass
            # p.add_trace(go.Scatter(x=fpr, y=tpr, hoverinfo='skip', mode='lines', line=dict(color=blue_color), showlegend=False,  opacity=0.1))
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = np.maximum(mean_tprs - std, 0)
    mean_rocauc = np.mean(roc_aucs).round(2)
    sd_rocauc = np.std(roc_aucs, ddof=1).round(2)

    if cohort_combos is None:
        p.add_trace(
            go.Scatter(
                x=base_fpr,
                y=tprs_lower,
                fill=None,
                line_color="gray",
                opacity=0.1,
                showlegend=False,
            )
        )
        p.add_trace(
            go.Scatter(
                x=base_fpr,
                y=tprs_upper,
                fill="tonexty",
                line_color="gray",
                opacity=0.1,
                name="±1 std. dev",
            )
        )
        hovertemplate = "Base FPR %{x:.2f} <br>%{text}"
        text = [
            "Upper TPR {:.2f} <br>Mean TPR {:.2f} <br>Lower TPR {:.2f}".format(u, m, l)
            for u, m, l in zip(tprs_upper, mean_tprs, tprs_lower)
        ]
        p.add_trace(
            go.Scatter(
                x=base_fpr,
                y=mean_tprs,
                text=text,
                hovertemplate=hovertemplate,
                hoverinfo="y+text",
                line=dict(color="black", width=2),
                name="Mean ROC\n(AUC = {:.2f}±{:.2f})".format(mean_rocauc, sd_rocauc),
            )
        )
        p.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                line=dict(color=red_color, dash="dash"),
                name="Chance",
            )
        )
    else:
        p.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                line=dict(color="black", dash="dash"),
                name="Chance",
            )
        )

    # Setting the figure layouts
    p.update_xaxes(showline=True, linewidth=1, linecolor="black")
    p.update_yaxes(showline=True, linewidth=1, linecolor="black")
    p.update_layout(
        autosize=True,
        width=700,
        height=700,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return p


# Prepare PR Curve
def plot_pr_curve_cv(pr_curve_results, class_ratio_test, cohort_combos=None):
    """
    Returns Plotly chart for Precision-Recall (PR) curve
    """
    precisions = []
    base_recall = np.linspace(0, 1, 101)
    pr_aucs = []
    p = go.Figure()

    for idx, (precision, recall, _) in enumerate(pr_curve_results):
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        if cohort_combos is not None:
            pr_df = pd.DataFrame(
                {
                    "recall": recall,
                    "precision": precision,
                    "train": cohort_combos[idx][0],
                    "test": cohort_combos[idx][1],
                }
            )
            text = "Train: {} <br>Test: {}".format(
                cohort_combos[idx][0], cohort_combos[idx][1]
            )
            hovertemplate = "Recall: %{x:.2f} <br>Precision: %{y:.2f}" + "<br>" + text
            p.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    hovertemplate=hovertemplate,
                    hoverinfo="all",
                    mode="lines",
                    name="Train on {}, Test on {}, AUC {:.2f}".format(
                        cohort_combos[idx][0], cohort_combos[idx][1], pr_auc
                    ),
                )
            )
        else:
            pass
            # p.add_trace(go.Scatter(x=recall, y=precision, hoverinfo='skip', mode='lines', line=dict(color=blue_color'), showlegend=False,  opacity=0.1))
        precision = np.interp(base_recall, recall, precision, period=100)
        precision[0] = 1.0
        precisions.append(precision)

    precisions = np.array(precisions)
    mean_precisions = precisions.mean(axis=0)
    std = precisions.std(axis=0)
    precisions_upper = np.minimum(mean_precisions + std, 1)
    precisions_lower = np.maximum(mean_precisions - std, 0)
    mean_prauc = np.mean(pr_aucs).round(2)
    sd_prauc = np.std(pr_aucs, ddof=1).round(2)

    if cohort_combos is None:
        p.add_trace(
            go.Scatter(
                x=base_recall,
                y=precisions_lower,
                fill=None,
                line_color="gray",
                opacity=0.1,
                showlegend=False,
            )
        )
        p.add_trace(
            go.Scatter(
                x=base_recall,
                y=precisions_upper,
                fill="tonexty",
                line_color="gray",
                opacity=0.2,
                name="±1 std. dev",
            )
        )
        hovertemplate = "Base Recall %{x:.2f} <br>%{text}"
        text = [
            "Upper Precision {:.2f} <br>Mean Precision {:.2f} <br>Lower Precision {:.2f}".format(
                u, m, l
            )
            for u, m, l in zip(precisions_upper, mean_precisions, precisions_lower)
        ]
        p.add_trace(
            go.Scatter(
                x=base_recall,
                y=mean_precisions,
                text=text,
                hovertemplate=hovertemplate,
                hoverinfo="y+text",
                line=dict(color="black", width=2),
                name="Mean PR\n(AUC = {:.2f}±{:.2f})".format(mean_prauc, sd_prauc),
            )
        )
        no_skill = np.mean(class_ratio_test)
        p.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[no_skill, no_skill],
                line=dict(color=red_color, dash="dash"),
                name="Chance",
            )
        )
    else:
        no_skill = np.mean(class_ratio_test)
        p.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[no_skill, no_skill],
                line=dict(color="black", dash="dash"),
                name="Chance",
            )
        )

    # Setting the figure layouts
    p.update_xaxes(showline=True, linewidth=1, linecolor="black")
    p.update_yaxes(showline=True, linewidth=1, range=[0, 1], linecolor="black")
    p.update_layout(
        autosize=True,
        width=700,
        height=700,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return p


# Generate dendograms for hier. clustering
def generate_dendrogram(
    matrix,
    labels,
    show_distances: bool = False,
    colorbar_title: str = "",
):
    """Generate Dendrogram."""

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(
        matrix,
        orientation="bottom",
        labels=labels,
        colorscale=[gray_color] * 8,
    )
    for i in range(len(fig["data"])):
        fig["data"][i]["yaxis"] = "y2"

    # Create side dendrogram
    dendro_side = ff.create_dendrogram(
        matrix, orientation="right", colorscale=[gray_color] * 8
    )
    for i in range(len(dendro_side["data"])):
        dendro_side["data"][i]["xaxis"] = "x2"

    # Add Side Dendrogram Data to Figure
    for data in dendro_side["data"]:
        fig.add_trace(data)

    # define dendro leaves
    dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
    dendro_leaves = list(map(int, dendro_leaves))

    # get heatmap data (z)
    if show_distances:
        # calculate distances
        heat_data = squareform(pdist(matrix))
    else:
        heat_data = matrix.values

    # arrange the heatmap data according to the dendrogram clustering
    heat_data = heat_data[dendro_leaves, :]
    heat_data = heat_data[:, dendro_leaves]

    heatmap = [
        go.Heatmap(
            x=dendro_leaves,
            y=dendro_leaves,
            z=heat_data,
            colorscale=[
                [0.0, blue_color],
                [0.5, "#ffffff"],
                [1.0, red_color],
            ],
            colorbar={"title": colorbar_title},
            hovertemplate=(
                "<b>Protein x:</b> %{x}<br><b>Protein y:</b> %{y}"
                "<extra>r = %{z:.2f}</extra>"
            ),
        )
    ]

    heatmap[0]["x"] = fig["layout"]["xaxis"]["tickvals"]
    heatmap[0]["y"] = dendro_side["layout"]["yaxis"]["tickvals"]
    for data in heatmap:
        fig.add_trace(data)

    # modify layout
    fig.update_layout(
        {
            "width": 800,
            "height": 800,
            "showlegend": False,
            "hovermode": "closest",
        }
    )

    # add labels to yaxis (needed for the hover)
    fig["layout"]["yaxis"]["ticktext"] = fig["layout"]["xaxis"]["ticktext"]
    fig["layout"]["yaxis"]["tickvals"] = fig["layout"]["xaxis"]["tickvals"]

    # modify axes
    params: dict = {
        "mirror": False,
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
        "ticks": "",
    }
    fig.update_layout(
        xaxis={"domain": [0.15, 1], **params},
        xaxis2={"domain": [0, 0.15], **params},
        yaxis={"domain": [0, 0.85], **params},
        yaxis2={"domain": [0.825, 0.975], **params},
    )

    return fig


# Perform EDA and Prepare their plots
def perform_EDA(state):
    """
    Perform EDA on the dataset by given method and return the chart
    """

    data = state.df_sub[state.proteins].astype("float").fillna(0.0)
    if state.eda_method == "Hierarchical clustering":
        data_to_be_correlated = data.iloc[:, state.data_range[0] : state.data_range[1]]
        corr = data_to_be_correlated.corr(method="pearson")
        labels = corr.columns
        p = generate_dendrogram(
            matrix=corr,
            labels=labels,
            colorbar_title="Pearson correlation coeff.",
        )

        p.update_layout(
            autosize=True,
            width=800,
            height=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor="rgba(255, 255, 255, 0)",
        )

    elif state.eda_method == "PCA":
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(data)
        components = pca.transform(data)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        pca_color = state.df_sub_y.replace(
            {True: ",".join(state.class_0), False: ",".join(state.class_1)}
        )
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        labels["color"] = state.target_column
        p = px.scatter(
            components,
            x=0,
            y=1,
            color=pca_color,
            labels=labels,
            hover_name=data.index,
        )

        # Show feature lines
        if hasattr(state, "pca_show_features") and (state.pca_show_features == True):
            for i, feature in enumerate(data.columns):
                p.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=loadings[i, 0],
                    y1=loadings[i, 1],
                )
                p.add_annotation(
                    x=loadings[i, 0],
                    y=loadings[i, 1],
                    ax=0,
                    ay=0,
                    xanchor="center",
                    yanchor="bottom",
                    text=feature,
                )

        # Tune other configs
        p.update_xaxes(showline=True, linewidth=1, linecolor="black")
        p.update_yaxes(showline=True, linewidth=1, linecolor="black")
        p.update_layout(
            autosize=True,
            width=700,
            height=500,
            xaxis_title="PCA 1",
            yaxis_title="PCA 2",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor="rgba(255, 255, 255, 0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

    return p
