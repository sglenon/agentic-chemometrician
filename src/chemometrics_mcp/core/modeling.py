"""Chemometrics modeling: CV-based training, metric computation, and figure data generation.

All models use scikit-learn. No matplotlib — figures are JSON data dicts.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

try:
    import xgboost as _xgb
except ImportError:
    _xgb = None

try:
    from chemometrics_mcp.core.interpretation import compute_shap_importance
except ImportError:
    compute_shap_importance = None

from chemometrics_contracts import AnalysisResult, InterpretationResult, ValidationWarning

_UNSUPERVISED_TASKS = {"unsupervised_exploration", "clustering"}


def make_cv_splitter(validation_strategy: str, y: np.ndarray | None = None, seed: int = 42):
    """Return a scikit-learn CV splitter from a strategy name string.

    Parameters
    ----------
    validation_strategy:
        One of ``"stratified_kfold_5"``, ``"stratified_kfold_3"``,
        ``"grouped_kfold_5"``, ``"loocv"``, or any other string (falls back
        to ``KFold(n_splits=5)``).
    y:
        Optional label array (not currently used but kept for future group support).
    seed:
        Random state for reproducible splits. Defaults to 42.
    """
    if validation_strategy == "stratified_kfold_5":
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if validation_strategy == "stratified_kfold_3":
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    if validation_strategy == "grouped_kfold_5":
        return KFold(n_splits=5, shuffle=True, random_state=seed)
    if validation_strategy == "loocv":
        return LeaveOneOut()
    return KFold(n_splits=5, shuffle=True, random_state=seed)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "confusion_matrix": cm.tolist(),
        "class_labels": [str(lbl) for lbl in labels],
    }


def _predicted_vs_actual_figure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "predicted_vs_actual": {
            "predicted": [float(v) for v in y_pred],
            "actual": [float(v) for v in y_true],
        }
    }


def run_cv_model(
    X: np.ndarray,
    y: np.ndarray | None,
    axis: np.ndarray | None,
    model_name: str,
    cv,
    task_name: str,
    preprocessing_applied: str,
    validation_strategy: str | None = None,
    enable_shap: bool = False,
) -> tuple[AnalysisResult, dict]:
    """Run a model with cross-validation and return (AnalysisResult, figure_data_dict).

    Parameters
    ----------
    X:
        2D feature array (n_samples, n_features).
    y:
        Label or target array; may be None for unsupervised models.
    axis:
        Spectral axis array; used for feature labelling in some models.
    model_name:
        One of ``"svm_rbf"``, ``"random_forest"``, ``"pca_lda"``, ``"pca"``,
        ``"kmeans"``, ``"plsr"``, ``"svr"``.
    cv:
        scikit-learn CV splitter (from ``make_cv_splitter``).
    task_name:
        Task type string (e.g. ``"binary_classification"``).
    preprocessing_applied:
        Name of the preprocessing method applied before this call.

    Returns
    -------
    tuple of (AnalysisResult, figure_data_dict).

    Raises
    ------
    ValueError
        For unknown model names or missing labels when required.
    """
    n_samples, n_features = X.shape

    # ------------------------------------------------------------------ svm_rbf
    if model_name == "svm_rbf":
        if y is None or task_name in _UNSUPERVISED_TASKS:
            raise ValueError("svm_rbf requires class labels")
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            decision_function_shape="ovr",
            random_state=42,
        )
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        metrics = _classification_metrics(y, y_pred)
        fig_data = _confusion_matrix_figure(y, y_pred)

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            clf.fit(X, y)
            shap_data = compute_shap_importance(
                clf, X, axis=axis, task_type="classification"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(y_pred.tolist()),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    # ------------------------------------------------------------ random_forest
    if model_name == "random_forest":
        if y is None or task_name in _UNSUPERVISED_TASKS:
            raise ValueError("random_forest requires class labels")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        metrics = _classification_metrics(y, y_pred)

        # Feature importances: fit on full data for importance extraction
        clf.fit(X, y)
        importances = clf.feature_importances_
        top20_idx = np.argsort(importances)[::-1][:20]
        top10_idx = np.argsort(importances)[::-1][:10]

        axis_list = axis.tolist() if axis is not None else None
        if axis_list is not None:
            top20_axis = [float(axis_list[i]) for i in top20_idx if i < len(axis_list)]
            selected = tuple(float(axis_list[i]) for i in top10_idx if i < len(axis_list))
        else:
            top20_axis = [int(i) for i in top20_idx]
            selected = tuple(int(i) for i in top10_idx)

        fig_data = {
            "feature_importances": [float(importances[i]) for i in top20_idx],
            "axis": top20_axis,
        }

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            shap_data = compute_shap_importance(
                clf, X, axis=axis, task_type="classification"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(y_pred.tolist()),
            selected_features=selected,
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    # --------------------------------------------------------------- pca_lda
    if model_name == "pca_lda":
        if y is None or task_name in _UNSUPERVISED_TASKS:
            raise ValueError("pca_lda requires class labels")
        n_pca = min(20, n_features, n_samples - 1)
        pipeline = Pipeline(
            [
                ("pca", PCA(n_components=n_pca)),
                ("lda", LinearDiscriminantAnalysis()),
            ]
        )
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        metrics = _classification_metrics(y, y_pred)

        # Per-fold accuracy for figure data
        pipeline.fit(X, y)
        n_lda = pipeline.named_steps["lda"].scalings_.shape[1]

        fig_data = {
            "cv_accuracy_per_fold": [],
            "n_lda_components": int(n_lda),
        }
        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(y_pred.tolist()),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
        )
        return result, fig_data

    # --------------------------------------------------------------- pca
    if model_name == "pca":
        n_components = min(10, n_features, n_samples)
        pca = PCA(n_components=n_components)
        pca.fit(X)
        evr = pca.explained_variance_ratio_.tolist()
        cumulative = [float(sum(evr[: i + 1])) for i in range(len(evr))]
        metrics = {
            "explained_variance_ratio_cumulative": float(sum(evr)),
            "n_components": int(n_components),
        }
        fig_data = {
            "explained_variance_ratio": [float(v) for v in evr],
            "cumulative_variance": cumulative,
            "n_components": int(n_components),
        }
        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=(),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
        )
        return result, fig_data

    # --------------------------------------------------------------- kmeans
    if model_name == "kmeans":
        n_clusters = len(set(y.tolist())) if y is not None else 3
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        km.fit(X)
        labels_pred = km.labels_
        cluster_sizes = [int((labels_pred == k).sum()) for k in range(n_clusters)]
        metrics = {
            "inertia": float(km.inertia_),
            "n_clusters": int(n_clusters),
        }
        fig_data = {"cluster_sizes": cluster_sizes}
        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(labels_pred.tolist()),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
        )
        return result, fig_data

    # --------------------------------------------------------------- plsr
    if model_name == "plsr":
        if y is None:
            raise ValueError("plsr requires a numeric target y")
        y_arr = np.array(y, dtype=float)
        n_components = min(10, n_features, n_samples - 1)
        plsr = PLSRegression(n_components=n_components)
        y_pred = cross_val_predict(plsr, X, y_arr, cv=cv).ravel()
        metrics = _regression_metrics(y_arr, y_pred)

        # selected_features: top-5 by absolute loading magnitude (first component)
        plsr.fit(X, y_arr)
        loadings = plsr.x_loadings_[:, 0]  # shape (n_features,)
        top5_idx = np.argsort(np.abs(loadings))[::-1][:5]
        if axis is not None:
            selected = tuple(float(axis[i]) for i in top5_idx if i < len(axis))
        else:
            selected = tuple(int(i) for i in top5_idx)

        fig_data = _predicted_vs_actual_figure(y_arr, y_pred)

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            shap_data = compute_shap_importance(
                plsr, X, axis=axis, task_type="regression"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(float(v) for v in y_pred),
            selected_features=selected,
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    # --------------------------------------------------------------- svr
    if model_name == "svr":
        if y is None:
            raise ValueError("svr requires a numeric target y")
        y_arr = np.array(y, dtype=float)
        svr = SVR(kernel="rbf", C=1.0, gamma="scale")
        y_pred = cross_val_predict(svr, X, y_arr, cv=cv).ravel()
        metrics = _regression_metrics(y_arr, y_pred)
        fig_data = _predicted_vs_actual_figure(y_arr, y_pred)

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            svr.fit(X, y_arr)
            shap_data = compute_shap_importance(
                svr, X, axis=axis, task_type="regression"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(float(v) for v in y_pred),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    # --------------------------------------------------------------- xgboost
    if model_name == "xgboost":
        if _xgb is None:
            raise ValueError("xgboost is not installed")
        if y is None or task_name in _UNSUPERVISED_TASKS:
            raise ValueError("xgboost requires class labels")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        clf = _xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        y_pred_enc = cross_val_predict(clf, X, y_enc, cv=cv)
        y_pred = le.inverse_transform(y_pred_enc)
        metrics = _classification_metrics(y, y_pred)

        clf.fit(X, y_enc)
        importances = clf.feature_importances_
        top20_idx = np.argsort(importances)[::-1][:20]
        top10_idx = np.argsort(importances)[::-1][:10]

        axis_list = axis.tolist() if axis is not None else None
        if axis_list is not None:
            top20_axis = [float(axis_list[i]) for i in top20_idx if i < len(axis_list)]
            selected = tuple(float(axis_list[i]) for i in top10_idx if i < len(axis_list))
        else:
            top20_axis = [int(i) for i in top20_idx]
            selected = tuple(int(i) for i in top10_idx)

        fig_data = {
            "feature_importances": [float(importances[i]) for i in top20_idx],
            "axis": top20_axis,
        }

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            shap_data = compute_shap_importance(
                clf, X, axis=axis, task_type="classification"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(y_pred.tolist()),
            selected_features=selected,
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    # --------------------------------------------------------------- xgboost_reg
    if model_name == "xgboost_reg":
        if _xgb is None:
            raise ValueError("xgboost is not installed")
        if y is None:
            raise ValueError("xgboost_reg requires a numeric target y")
        y_arr = np.array(y, dtype=float)
        reg = _xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="rmse",
        )
        y_pred = cross_val_predict(reg, X, y_arr, cv=cv).ravel()
        metrics = _regression_metrics(y_arr, y_pred)
        fig_data = _predicted_vs_actual_figure(y_arr, y_pred)

        interpretation_results: list[InterpretationResult] = []
        if enable_shap and compute_shap_importance is not None:
            reg.fit(X, y_arr)
            shap_data = compute_shap_importance(
                reg, X, axis=axis, task_type="regression"
            )
            interpretation_results.append(
                InterpretationResult(
                    model_name=model_name,
                    method="shap",
                    feature_scores=tuple(shap_data["feature_scores"]),
                    top_features=tuple(shap_data["top_features"]),
                )
            )

        result = AnalysisResult(
            task_name=task_name,
            model_name=model_name,
            preprocessing=(preprocessing_applied,),
            metrics=metrics,
            predictions=tuple(float(v) for v in y_pred),
            selected_features=(),
            figures=(),
            validation_strategy=validation_strategy,
            interpretation_results=tuple(interpretation_results),
        )
        return result, fig_data

    raise ValueError(f"Unknown model: {model_name!r}")


def run_cv_model_multi_seed(
    X: np.ndarray,
    y: np.ndarray | None,
    axis: np.ndarray | None,
    model_name: str,
    task_name: str,
    preprocessing_applied: str,
    validation_strategy: str | None = None,
    n_seeds: int = 3,
    base_seed: int = 42,
    group_column: str | None = None,
    enable_shap: bool = False,
) -> list[tuple[AnalysisResult, dict]]:
    """Run the same model with multiple random seeds for split-instability analysis.

    Parameters
    ----------
    X, y, axis, model_name, task_name, preprocessing_applied, validation_strategy:
        Same as :func:`run_cv_model`.
    n_seeds:
        Number of seeds to run. Defaults to 3.
    base_seed:
        Starting seed; subsequent runs use ``base_seed + i``. Defaults to 42.
    group_column:
        Reserved for future grouped CV support (currently unused).

    Returns
    -------
    List of ``(AnalysisResult, figure_data_dict)`` tuples, one per seed.
    """
    results: list[tuple[AnalysisResult, dict]] = []
    for i in range(n_seeds):
        seed = base_seed + i
        cv = make_cv_splitter(
            validation_strategy or "stratified_kfold_5", y, seed=seed
        )
        result, fig_data = run_cv_model(
            X, y, axis, model_name, cv, task_name, preprocessing_applied, validation_strategy, enable_shap
        )
        results.append((result, fig_data))
    return results
