"""
Main pipeline for uncertainty explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split

from .conformal.cqr_predictor import CQRConformalPredictor
from .conformal.crepes_classifier import (
    ClassificationConformalMethod,
    CrepesConformalClassifier,
)
from .conformal.crepes_predictor import (
    ConformalMethod,
    CrepesConformalPredictor,
)
from .explainability.lime_explainer import LimeUncertaintyExplainer
from .explainability.pdp_explainer import PDPUncertaintyExplainer
from .explainability.shap_explainer import ShapUncertaintyExplainer
from .plots import (
    generate_lime_plots,
    generate_pdp_plots,
    generate_shap_plots,
)
from .protocols import (
    ConformalPredictorProtocol,
    UncertaintyExplainerProtocol,
)
from .uncertainty.metrics import (
    UncertaintyMetric,
    is_classifier_predictor,
)

XAIMethod = Literal["shap", "pdp", "lime"]
TaskKind = Literal["auto", "regression", "classification"]

VALID_PLOT_KINDS = ("beeswarm", "bar", "waterfall", "summary")
VALID_PDP_PLOT_KINDS = ("pdp", "ice", "pdp_ice", "pdp_2d", "importance")
VALID_LIME_PLOT_KINDS = ("local",)

REGRESSION_METRICS = ("width", "lower", "upper", "midpoint")
CLASSIFICATION_METRICS = ("set_size", "credibility", "confidence")


@dataclass
class ExplanationResult:
    """Result of explain() for regression.

    Attributes
    ----------
    lower : np.ndarray
        Lower prediction bounds.
    upper : np.ndarray
        Upper prediction bounds.
    interval_width : np.ndarray
        Width of each prediction interval.
    explanation_values : Any
        Explanation object (e.g. shap.Explanation, lime output).
    """

    lower: np.ndarray
    upper: np.ndarray
    interval_width: np.ndarray
    explanation_values: Any


@dataclass
class ClassificationExplanationResult:
    """Result of explain() for classification.

    Attributes
    ----------
    prediction_set : np.ndarray of shape (n_samples, n_classes), dtype=bool
        Conformal prediction sets — True at ``[i, k]`` means class ``k``
        is included in the set for sample ``i``.
    p_values : np.ndarray of shape (n_samples, n_classes)
        Conformal p-values per class.
    set_size : np.ndarray of shape (n_samples,)
        Size of the prediction set for each sample.
    classes : np.ndarray of shape (n_classes,)
        Class labels in the order used by ``prediction_set`` and ``p_values``.
    explanation_values : Any
        Explainer output (shap.Explanation, LIMEExplanation, PDPExplanation).
    """

    prediction_set: np.ndarray
    p_values: np.ndarray
    set_size: np.ndarray
    classes: np.ndarray
    explanation_values: Any


def _detect_task(
    model,
    conformal_predictor,
    task: TaskKind,
) -> Literal["regression", "classification"]:
    """Resolve ``task="auto"`` from the model / predictor."""

    if task != "auto":
        return task
    if conformal_predictor is not None:
        return (
            "classification" if is_classifier_predictor(conformal_predictor)
            else "regression"
        )
    if model is not None and is_classifier(model):
        return "classification"
    return "regression"


class UncertaintyExplanationPipeline:
    """
    Pipeline integrating conformal prediction and uncertainty explanation.

    Supports both regression and classification (selected via ``task``).

    Three XAI methods are available, selectable via ``xai_method``:

    - ``"shap"`` — SHAP-based explanation (default)
    - ``"pdp"``  — Partial Dependence Plot explanation
    - ``"lime"`` — LIME-based explanation

    Examples
    --------
    Regression::

        from sklearn.ensemble import RandomForestRegressor
        pipeline = UncertaintyExplanationPipeline(model=RandomForestRegressor())
        pipeline.fit(X_train, y_train)
        result = pipeline.explain(X_test)

    Classification::

        from sklearn.ensemble import RandomForestClassifier
        pipeline = UncertaintyExplanationPipeline(
            model=RandomForestClassifier(),
            task="classification",
            uncertainty_metric="set_size",
        )
        pipeline.fit(X_train, y_train)
        result = pipeline.explain(X_test)
        print(result.prediction_set)   # (n, n_classes) bool
        print(result.set_size)         # (n,)
    """

    def __init__(
        self,
        model=None,
        confidence: float = 0.9,
        task: TaskKind = "auto",
        conformal_method: (
            ConformalMethod | ClassificationConformalMethod | Literal["cqr"]
            | None
        ) = None,
        xai_method: XAIMethod = "shap",
        uncertainty_metric: UncertaintyMetric | None = None,
        n_lime_samples: int = 5000,
        random_state: int | None = None,
        lower_model=None,
        upper_model=None,
        conformal_predictor: ConformalPredictorProtocol | None = None,
        explainer: UncertaintyExplainerProtocol | None = None,
        fast_shap: bool = True,
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        model
            sklearn-compatible regressor or classifier. Required for crepes
            methods. Ignored when ``conformal_method="cqr"`` or when a custom
            ``conformal_predictor`` is provided.

        confidence : float
            Confidence level for conformal prediction 

        task : {"auto", "regression", "classification"}
            Task type. ``"auto"`` infers from the model / conformal_predictor.

        conformal_method : str, optional
            Conformal method.

            - Regression: ``"standard"``, ``"normalized"`` (default),
              ``"mondrian"``, ``"normalized_mondrian"`` (crepes), or ``"cqr"``.
            - Classification: ``"standard"`` (default), ``"class_cond"``,
              ``"mondrian"``.

            Ignored if ``conformal_predictor`` is provided.

        xai_method : {"shap", "pdp", "lime"}
            Explainability method to use. Ignored if ``explainer`` is provided.

        uncertainty_metric : str, optional
            Which scalar to explain. Defaults depend on ``task``:

            - Regression: ``"width"`` (interval width).
              Other options: ``"lower"``, ``"upper"``, ``"midpoint"``.
            - Classification: ``"set_size"`` (size of prediction set).
              Other options: ``"credibility"``, ``"confidence"``.

            Ignored if ``explainer`` is provided.

        n_lime_samples : int
            Number of LIME perturbations per sample. Ignored when
            ``xai_method != "lime"``.

        random_state : int, optional
            Seed for the auto calibration split and LIME perturbation sampler.

        lower_model, upper_model
            Quantile regressors for CQR. Required when
            ``conformal_method="cqr"`` (regression only).

        conformal_predictor : optional
            Custom conformal predictor (regressor or classifier). Overrides
            ``conformal_method`` when provided.

        explainer : optional
            Custom explainer instance. Overrides ``xai_method`` when provided.

        fast_shap : bool
            When ``xai_method="shap"``, use exact TreeSHAP on the component
            models for metrics that are affine in them (e.g. CQR interval
            width, which is the difference of the two quantile models). Falls
            back to the generic explainer whenever that does not apply. Set to
            ``False`` to always use the generic path.
        """

        self.confidence = confidence
        self.xai_method = xai_method
        self.random_state = random_state

        # Resolve task
        self.task = _detect_task(model, conformal_predictor, task)

        # Resolve metric default per task
        if uncertainty_metric is None:
            uncertainty_metric = (
                "set_size" if self.task == "classification" else "width"
            )
        self._validate_metric(uncertainty_metric, self.task)
        self.uncertainty_metric = uncertainty_metric

        # Resolve conformal method default per task
        if conformal_method is None:
            conformal_method = (
                "standard" if self.task == "classification" else "normalized"
            )

        # Build conformal predictor
        if conformal_predictor is not None:
            self.cp = conformal_predictor
        elif self.task == "classification":
            if model is None:
                raise ValueError(
                    "task='classification' requires a 'model' (sklearn-"
                    "compatible classifier with predict_proba) or a custom "
                    "'conformal_predictor'."
                )
            if conformal_method not in (
                "standard", "class_cond", "mondrian"
            ):
                raise ValueError(
                    f"conformal_method='{conformal_method}' is not valid for "
                    "classification. Choose from 'standard', 'class_cond', "
                    "'mondrian'."
                )
            self.cp = CrepesConformalClassifier(
                model,
                method=conformal_method,
                random_state=random_state,
            )
        elif conformal_method == "cqr":
            if lower_model is None or upper_model is None:
                raise ValueError(
                    "conformal_method='cqr' requires both 'lower_model' "
                    "and 'upper_model'."
                )
            self.cp = CQRConformalPredictor(lower_model, upper_model)
        elif model is not None:
            self.cp = CrepesConformalPredictor(
                model,
                method=conformal_method,
            )
        else:
            raise ValueError(
                "Either 'model' or 'conformal_predictor' must be provided, "
                "or set conformal_method='cqr' with 'lower_model' and "
                "'upper_model'."
            )

        # Build explainer (regression and classification share the same
        # explainers — they both consume a scalar uncertainty function)
        if explainer is not None:
            self.explainer = explainer
        elif xai_method == "pdp":
            self.explainer = PDPUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                metric=self.uncertainty_metric,
            )
        elif xai_method == "shap":
            self.explainer = ShapUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                metric=self.uncertainty_metric,
                fast_path=fast_shap,
            )
        elif xai_method == "lime":
            self.explainer = LimeUncertaintyExplainer(
                cp=self.cp,
                confidence=self.confidence,
                n_lime_samples=n_lime_samples,
                random_state=random_state,
                metric=self.uncertainty_metric,
            )
        else:
            raise ValueError(
                f"Unknown xai_method '{xai_method}'. "
                "Choose from 'shap', 'pdp', or 'lime'."
            )

        self._is_fitted = False
        self._explainer_kwargs = None
        self._feature_names = None

    def fit(
        self,
        X_train,
        y_train,
        X_calib=None,
        y_calib=None,
        calib_size: float = 0.2,
        X_background=None,
        random_state: int | None = None,
    ):
        """
        Fit conformal predictor.

        If X_calib and y_calib are not provided, they are
        split automatically from X_train using calib_size.

        Parameters
        ----------
        X_train : np.ndarray or DataFrame
        y_train : np.ndarray or Series
        X_calib : np.ndarray or DataFrame, optional
        y_calib : np.ndarray or Series, optional
        calib_size : float
            Fraction of X_train to use for calibration when X_calib is not
            provided.
        X_background : np.ndarray, optional
            Background data for the explainer. Defaults to X_calib.
        random_state : int, optional
            Seed for the auto calibration split. Overrides the pipeline-level
            ``random_state`` for this call only.
        """

        # Extract feature names from DataFrame before converting
        if hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
            if hasattr(self.explainer, "feature_names"):
                self.explainer.feature_names = self._feature_names

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_calib is not None:
            X_calib = np.asarray(X_calib)
        if y_calib is not None:
            y_calib = np.asarray(y_calib)

        # Auto-split if calibration set not provided.
        # Stratify on y for classification to avoid missing classes in calib.
        if X_calib is None or y_calib is None:
            seed = random_state if random_state is not None else self.random_state
            stratify = y_train if self.task == "classification" else None
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, test_size=calib_size,
                random_state=seed,
                stratify=stratify,
            )

        # Strip feature names before fitting. SHAP/LIME later call predict()
        # with numpy arrays; sklearn emits a warning on every such call when
        # the estimator was fitted with a DataFrame.
        X_train = np.asarray(X_train)
        X_calib = np.asarray(X_calib)
        y_train = np.asarray(y_train)
        y_calib = np.asarray(y_calib)

        self.cp.fit(
            X_train,
            y_train,
            X_calib,
            y_calib,
        )

        self._X_background = (
            np.asarray(X_background) if X_background is not None
            else X_calib
        )
        self._is_fitted = True
        self._explainer_kwargs = None

    def predict(
        self,
        X,
        confidence: float | None = None,
    ):
        """
        Generate conformal predictions.

        Returns
        -------
        Regression
            ``(lower, upper)`` — tuple of arrays.
        Classification
            ``prediction_set`` — array of shape ``(n_samples, n_classes)``,
            dtype=bool.
        """

        self._check_is_fitted()
        self._check_X(X)

        conf = self.confidence if confidence is None else confidence
        X_arr = np.asarray(X)

        if self.task == "classification":
            return self.cp.predict_set(X_arr, confidence=conf)
        return self.cp.predict(X_arr, confidence=conf)

    def explain(
        self,
        X,
        show_plots: bool = True,
        plot_kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
        figsize: tuple[float, float] | None = None,
        **explainer_kwargs,
    ):
        """
        Explain uncertainty for X.

        Parameters
        ----------
        X : np.ndarray
        show_plots : bool
            Whether to display plots.
        plot_kind : str or list of str, optional
            For SHAP: ``"beeswarm"``, ``"bar"``, ``"waterfall"``, ``"summary"``.
            For PDP:  ``"pdp"``, ``"ice"``, ``"pdp_ice"``, ``"pdp_2d"``, ``"importance"``.
            For LIME: ``"local"``.
            Defaults to method-specific defaults when ``None``.
        waterfall_index : int
            Sample index for SHAP waterfall plot or LIME local plot.
        figsize : tuple of (width, height), optional
            Figure size in inches forwarded to the plot function.
        **explainer_kwargs
            Passed to explainer.fit().

        Returns
        -------
        ExplanationResult or ClassificationExplanationResult
            Depending on ``self.task``.
        """

        self._check_is_fitted()
        self._check_X(X)
        if isinstance(self.explainer, PDPUncertaintyExplainer) and np.asarray(X).shape[0] == 1:
            raise ValueError(
                "PDP does not support single-sample (local) explanations. "
                "Use xai_method='shap' or 'lime' instead."
            )
        features_kw = explainer_kwargs.get("features") or []
        pairs = [f for f in features_kw if isinstance(f, tuple)]
        is_pdp = isinstance(self.explainer, PDPUncertaintyExplainer)

        if plot_kind is None and pairs and is_pdp:
            plot_kind = ["pdp_2d", "pdp"] 
        if plot_kind is not None:
            self._resolve_plot_kinds(plot_kind, X=X)
            kinds = [plot_kind] if isinstance(plot_kind, str) else plot_kind
            if "pdp_2d" in kinds and not pairs:
                raise ValueError(
                    "plot_kind='pdp_2d' requires at least one feature pair as a tuple, "
                    "e.g. features=[(0, 1)] or features=[0, 1, (0, 1)]."
                )

        # Rebuild explainer if kwargs changed
        if self._explainer_kwargs != explainer_kwargs:
            self.explainer.fit(
                self._X_background,
                **explainer_kwargs,
            )
            self._explainer_kwargs = explainer_kwargs

        explanation_values = self.explainer.explain(X)

        X_arr = np.asarray(X)

        if self.task == "classification":
            prediction_set = self.cp.predict_set(X_arr, confidence=self.confidence)
            p_values = self.cp.predict_p(X_arr)
            set_size = prediction_set.sum(axis=1)
            classes = np.asarray(self.cp.classes_)

            result = ClassificationExplanationResult(
                prediction_set=prediction_set,
                p_values=p_values,
                set_size=set_size,
                classes=classes,
                explanation_values=explanation_values,
            )
        else:
            lower, upper = self.cp.predict(X_arr, confidence=self.confidence)
            width = upper - lower
            result = ExplanationResult(
                lower=lower,
                upper=upper,
                interval_width=width,
                explanation_values=explanation_values,
            )

        if show_plots:
            self.plot(
                explanation_values,
                X=X,
                kind=plot_kind,
                waterfall_index=waterfall_index,
                figsize=figsize,
            )

        return result

    explain_uncertainty = explain

    def plot(
        self,
        explanation_values,
        X=None,
        kind: str | list[str] | None = None,
        waterfall_index: int | None = None,
        figsize: tuple[float, float] | None = None,
        result=None,
    ):
        """
        Generate plot(s) from existing explanation results.

        Parameters
        ----------
        figsize : tuple of (width, height), optional
            Figure size in inches forwarded to the underlying plot
            function. When ``None``, each plot function applies its
            own default.
        """

        kinds = self._resolve_plot_kinds(kind, X=X)

        if isinstance(self.explainer, PDPUncertaintyExplainer):
            generate_pdp_plots(
                explanation_values,
                kinds=kinds,
                feature_names=self._feature_names,
                figsize=figsize,
            )
        elif isinstance(self.explainer, LimeUncertaintyExplainer):
            generate_lime_plots(
                explanation_values,
                kinds=kinds,
                sample_index=waterfall_index,
                figsize=figsize,
            )
        else:
            shap_kwargs = {"figsize": figsize} if figsize is not None else {}
            generate_shap_plots(
                explanation_values,
                X,
                kinds=kinds,
                feature_names=self._feature_names,
                waterfall_index=waterfall_index,
                **shap_kwargs,
            )


    def _validate_metric(self, metric, task):
        if task == "classification" and metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"uncertainty_metric='{metric}' is not valid for "
                f"classification. Choose from {CLASSIFICATION_METRICS}."
            )
        if task == "regression" and metric not in REGRESSION_METRICS:
            raise ValueError(
                f"uncertainty_metric='{metric}' is not valid for "
                f"regression. Choose from {REGRESSION_METRICS}."
            )

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline is not fitted. Call fit() first."
            )

    def _check_X(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"X must be 2D, got shape {X_arr.shape}"
            )

    def _resolve_plot_kinds(
        self,
        kind: str | list[str] | None,
        X=None,
    ) -> list[str]:
        if isinstance(self.explainer, PDPUncertaintyExplainer):
            valid = VALID_PDP_PLOT_KINDS
        elif isinstance(self.explainer, LimeUncertaintyExplainer):
            valid = VALID_LIME_PLOT_KINDS
        else:
            valid = VALID_PLOT_KINDS
        if kind is None:
            if (
                isinstance(self.explainer, ShapUncertaintyExplainer)
                and X is not None
                and np.asarray(X).shape[0] == 1
            ):
                return ["waterfall"]
            return None  # each plot function applies its own defaults
        if isinstance(kind, str):
            kind = [kind]
        for k in kind:
            if k not in valid:
                raise ValueError(
                    f"Unknown plot kind '{k}'. "
                    f"Choose from {valid}"
                )
        return kind