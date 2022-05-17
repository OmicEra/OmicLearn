# Main
import numpy as np
import pandas as pd
import streamlit as st

# Sklearn
import sklearn
import sklearn.metrics as metrics
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import svm, tree, linear_model, neighbors, ensemble
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, SelectKBest
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    QuantileTransformer,
    PowerTransformer,
)

# Define base metrics to be used
scores = ["accuracy", "roc_auc", "precision", "recall", "f1", "balanced_accuracy"]
scorer_dict = {}
scorer_dict = {metric: metric + "_score" for metric in scores}
scorer_dict = {key: getattr(metrics, metric) for key, metric in scorer_dict.items()}


@st.cache(persist=True)
def transform_dataset(subset, additional_features, proteins):
    """
    Transforms data with label encoder
    """
    transformed_columns = []
    for _ in additional_features:
        if subset[_].dtype in [np.dtype("O"), np.dtype("str")]:
            le = LabelEncoder()
            transformed = le.fit_transform(subset[_])
        else:
            transformed = subset[_]
        transformed_columns.append(
            pd.DataFrame(transformed, columns=[_], index=subset[_].index)
        )

    if len(transformed_columns) > 1:
        transformed = pd.concat(transformed_columns, axis=1)
    elif len(transformed_columns) == 1:
        transformed = transformed_columns[0]
    else:
        transformed = []

    # Join with proteins
    protein_features = subset[proteins].astype("float")

    if len(transformed) >= 1 and len(protein_features) >= 1:
        X = pd.concat([protein_features, transformed], axis=1)
    else:
        if len(transformed) == 0:
            X = protein_features
        elif len(protein_features) == 0:
            X = transformed
        else:
            pass
    return X


def normalize_dataset(X, normalization, normalization_params):
    """
    Normalize/Scale data with scalers
    """

    class scaler_:
        def transform(self, x):
            return x

        def fit(self, x):
            pass

        def set_params(self, x):
            pass

    if normalization == "None":
        scaler = scaler_()
    elif normalization == "StandardScaler":
        scaler = StandardScaler()
    elif normalization == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif normalization == "RobustScaler":
        scaler = RobustScaler()
    elif normalization == "PowerTransformer":
        scaler = PowerTransformer()
        scaler.set_params(**normalization_params)
    elif normalization == "QuantileTransformer":
        scaler = QuantileTransformer()
        scaler.set_params(**normalization_params)
    else:
        raise NotImplementedError(
            f"Normalization method {normalization} not implemented"
        )

    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index), scaler


def select_features(feature_method, X, y, max_features, n_trees, random_state):
    """
    Returns the features and their imp. attributes based on the given method and params
    """
    if feature_method == "ExtraTrees":
        clf = ensemble.ExtraTreesClassifier(
            n_estimators=n_trees, random_state=random_state
        )
        clf = clf.fit(X.fillna(0), y)
        feature_importance = clf.feature_importances_
        top_sortindex = np.argsort(feature_importance)[::-1]
        p_values = np.empty(len(feature_importance))
        p_values[:] = np.nan

    elif "k-best" in feature_method:
        if feature_method == "k-best (mutual_info_classif)":
            clf = SelectKBest(mutual_info_classif, max_features)
        elif feature_method == "k-best (f_classif)":
            clf = SelectKBest(f_classif, max_features)
        elif feature_method == "k-best (chi2)":
            clf = SelectKBest(chi2, max_features)
        else:
            raise NotImplementedError(
                "Feature method {} not implemented.".format(feature_method)
            )
        clf = clf.fit(X.fillna(0), y)
        feature_importance = clf.scores_
        p_values = clf.pvalues_
        if p_values is None:
            p_values = np.empty(len(feature_importance))
            p_values[:] = np.nan
        top_sortindex = np.argsort(feature_importance)[::-1]

    elif feature_method == "None":
        max_features = len(X.columns)
        top_sortindex = np.arange(len(X.columns))
        p_values = np.zeros(len(X.columns))
        feature_importance = np.zeros(len(X.columns))
    else:
        raise NotImplementedError("Method {} not implemented.".format(feature_method))

    top_features = X.columns[top_sortindex][:max_features][::-1].tolist()
    top_features_importance = feature_importance[top_sortindex][:max_features][::-1]
    top_features_pvalues = p_values[top_sortindex][:max_features][::-1]

    return top_features, top_features_importance, top_features_pvalues


def impute_nan(X, missing_value, random_state):
    """
    Missing value imputation
    """

    class imputer_:
        def transform(self, x):
            return x

        def fit(self, x):
            pass

    X = X[X.columns[~X.isnull().all()]]  # Remove columns w only nans
    if missing_value == "Zero":
        imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    elif missing_value == "Mean":
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    elif missing_value == "Median":
        imp = SimpleImputer(missing_values=np.nan, strategy="median")
    elif missing_value == "None":
        imp = imputer_()
    elif missing_value == "KNNImputer":
        imp = KNNImputer()
    else:
        raise NotImplementedError("Method {} not implemented".format(missing_value))

    imp.fit(X)
    X = pd.DataFrame(imp.transform(X), columns=X.columns)

    return X, imp


def return_classifier(classifier, classifier_params):
    """
    Returns classifier object based on name
    """
    # Max Features parameter for RandomForest and DecisionTree
    cp = classifier_params.copy()
    if classifier in ["LogisticRegression", "KNeighborsClassifier", "RandomForest"]:
        cp["n_jobs"] = -1

    if classifier == "LinearSVC":
        cv_generator = cp["cv_generator"]
    else:
        cv_generator = None

    if classifier == "XGBoost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(use_label_encoder=False, eval_metric="error")
    elif classifier == "LogisticRegression":
        clf = linear_model.LogisticRegression()
    elif classifier == "KNeighborsClassifier":
        del cp["random_state"]
        clf = neighbors.KNeighborsClassifier()
    elif classifier == "RandomForest":
        clf = ensemble.RandomForestClassifier()
    elif classifier == "DecisionTree":
        clf = tree.DecisionTreeClassifier()
    elif classifier == "AdaBoost":
        clf = ensemble.AdaBoostClassifier()
    elif classifier == "LinearSVC":
        del cp["cv_generator"]
        clf = svm.LinearSVC()

    clf.set_params(**cp)
    return clf, cv_generator


def perform_cross_validation(state, cohort_column=None):
    """
    Performs cross-validation
    """
    clf, cv_generator = return_classifier(state.classifier, state.classifier_params)

    if state.cv_method == "RepeatedStratifiedKFold":
        cv_alg = RepeatedStratifiedKFold(
            n_splits=state.cv_splits,
            n_repeats=state.cv_repeats,
            random_state=state.random_state,
        )
    elif state.cv_method == "StratifiedKFold":
        cv_alg = StratifiedKFold(
            n_splits=state.cv_splits, shuffle=True, random_state=state.random_state
        )
    elif state.cv_method == "StratifiedShuffleSplit":
        cv_alg = StratifiedShuffleSplit(
            n_splits=state.cv_splits, random_state=state.random_state
        )
    else:
        raise NotImplementedError("This CV method is not implemented")

    _cv_results = {}
    _cv_curves = {}

    # Initialize reporting dict with empty lists
    for _ in ["num_feat", "n_obs", "n_class_0", "n_class_1", "class_ratio"]:
        for x in ["_train", "_test"]:
            _cv_results[_ + x] = []

    for _ in [
        "pr_auc",
        "roc_curves_",
        "pr_curves_",
        "y_hats_",
        "feature_importances_",
        "features_",
    ]:
        _cv_curves[_] = []

    for metric_name, metric_fct in scorer_dict.items():
        _cv_results[metric_name] = []
    _cv_results["pr_auc"] = []  # ADD pr_auc manually

    X = state.X
    y = state.y

    if cohort_column is not None:
        cohorts = state.X_cohort.unique().tolist()
        cohort_combos = []
        cohort_combo_names = []

        indexer = np.arange(len(X))
        for c_1 in cohorts:
            for c_2 in cohorts:
                if c_1 != c_2:
                    cohort_combos.append(
                        (indexer[state.X_cohort == c_1], indexer[state.X_cohort == c_2])
                    )
                    cohort_combo_names.append((c_1, c_2))

        iterator = cohort_combos
        cohort_combo_names_ = []
    else:
        iterator = cv_alg.split(X, y)

    X = X[state.features]
    for i, (train_index, test_index) in enumerate(iterator):
        # Missing value imputation
        X_train, imputer = impute_nan(
            X.iloc[train_index], state.missing_value, state.random_state
        )
        cols = X_train.columns  # Columns could be removed bc of nan
        X_test = pd.DataFrame(imputer.transform(X.iloc[test_index][cols]), columns=cols)

        # Normalization of data
        X_train, scaler = normalize_dataset(
            X_train, state.normalization, state.normalization_params
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Define y
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        skip = False
        if cohort_column is not None:
            if len(set(y_train)) == 1:
                st.info(
                    f"Only 1 class present in cohort {cohort_combo_names[i][0]}. Skipping training on {cohort_combo_names[i][0]} and predicting on {cohort_combo_names[i][1]}."
                )
                skip = True
            if len(set(y_test)) == 1:
                st.info(
                    f"Only 1 class present in cohort {cohort_combo_names[i][1]}. Skipping training on {cohort_combo_names[i][0]} and predicting on {cohort_combo_names[i][1]}."
                )
                skip = True
            if not skip:
                cohort_combo_names_.append(cohort_combo_names[i])

        if not skip:
            # Feature selection
            features_, feature_importance_, p_values = select_features(
                state.feature_method,
                X_train,
                y_train,
                state.max_features,
                state.n_trees,
                state.random_state,
            )

            X_train = X_train[features_]
            X_test = X_test[features_]

            # Fitting and predicting, and calculating prediction probabilities
            if state.classifier == "LinearSVC":
                # Since LinearSVC does not have `predict_proba()`
                from sklearn.calibration import CalibratedClassifierCV

                calibrated_clf = CalibratedClassifierCV(clf, cv=cv_generator)
                calibrated_clf.fit(X_train, y_train)
                y_pred = calibrated_clf.predict(X_test)
                y_pred_proba = calibrated_clf.predict_proba(X_test)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)

            # Feature importances received from classifier
            if state.classifier == "LogisticRegression":
                feature_importance = np.abs(clf.coef_[0])
            elif state.classifier == "LinearSVC":
                coef_avg = 0
                for j in calibrated_clf.calibrated_classifiers_:
                    coef_avg = coef_avg + j.base_estimator.coef_
                coef_avg = coef_avg / len(calibrated_clf.calibrated_classifiers_)
                feature_importance = coef_avg[0]
            elif state.classifier in [
                "AdaBoost",
                "RandomForest",
                "DecisionTree",
                "XGBoost",
            ]:
                feature_importance = clf.feature_importances_
            else:
                # Not implemented st.warning() for `KNeighborsClassifier`.
                feature_importance = None

            # ROC CURVE
            fpr, tpr, cutoffs = roc_curve(y_test, y_pred_proba[:, 1])

            # PR CURVE
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])

            for metric_name, metric_fct in scorer_dict.items():
                if metric_name == "roc_auc":
                    _cv_results[metric_name].append(
                        metric_fct(y_test, y_pred_proba[:, 1])
                    )
                elif metric_name in ["precision", "recall", "f1"]:
                    _cv_results[metric_name].append(
                        metric_fct(y_test, y_pred, zero_division=0)
                    )
                else:
                    _cv_results[metric_name].append(metric_fct(y_test, y_pred))

            # Results of Cross Validation
            _cv_results["num_feat_train"].append(X_train.shape[-1])
            _cv_results["n_obs_train"].append(len(y_train))
            _cv_results["n_class_0_train"].append(np.sum(y_train))
            _cv_results["n_class_1_train"].append(np.sum(~y_train))
            _cv_results["class_ratio_train"].append(np.sum(y_train) / len(y_train))
            _cv_results["num_feat_test"].append(X_test.shape[-1])
            _cv_results["n_obs_test"].append(len(y_test))
            _cv_results["n_class_0_test"].append(np.sum(y_test))
            _cv_results["n_class_1_test"].append(np.sum(~y_test))
            _cv_results["class_ratio_test"].append(np.sum(y_test) / len(y_test))
            _cv_results["pr_auc"].append(
                auc(recall, precision)
            )  # ADD PR Curve AUC Score
            _cv_curves["pr_auc"].append(
                auc(recall, precision)
            )  # ADD PR Curve AUC Score
            _cv_curves["roc_curves_"].append((fpr, tpr, cutoffs))
            _cv_curves["pr_curves_"].append((precision, recall, _))
            _cv_curves["y_hats_"].append((y_test.values, y_pred))

            if feature_importance is not None:
                _cv_curves["feature_importances_"].append(
                    dict(zip(X_train.columns.tolist(), feature_importance))
                )
            else:
                _cv_curves["feature_importances_"] = None

        if cohort_column is not None:
            state.bar.progress((i + 1) / len(cohort_combos))
        else:
            if state.cv_method == "RepeatedStratifiedKFold":
                state.bar.progress((i + 1) / (state.cv_splits * state.cv_repeats))
            else:
                state.bar.progress((i + 1) / (state.cv_splits))

        if cohort_column is not None:
            _cv_curves["cohort_combos"] = cohort_combo_names_

    return _cv_results, _cv_curves


def calculate_cm(y_test, y_pred):
    """
    Calculate confusion matrix
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i] == True:
            tp += 1
        if y_pred[i] == True and y_test[i] == False:
            fp += 1
        if y_test[i] == y_pred[i] == False:
            tn += 1
        if y_pred[i] == False and y_test[i] == True:
            fn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    return (tp, fp, tn, fn), (tpr, fpr, tnr, fnr)
