"""uxplain: explainability for conformal prediction uncertainty."""

__version__ = "0.2.3"

from uxplain.conformal.cqr_predictor import CQRConformalPredictor
from uxplain.conformal.crepes_classifier import (
    ClassificationConformalMethod,
    CrepesConformalClassifier,
)
from uxplain.conformal.crepes_predictor import (
    ConformalMethod,
    CrepesConformalPredictor,
)
from uxplain.explainability.lime_explainer import (
    LIMEExplanation,
    LimeUncertaintyExplainer,
)
from uxplain.explainability.pdp_explainer import (
    PDPExplanation,
    PDPUncertaintyExplainer,
)
from uxplain.explainability.shap_explainer import ShapUncertaintyExplainer
from uxplain.plotting import (
    STYLE,
    ice_curves,
    lime_local,
    pdp_curve,
    pdp_importance,
    pdp_interaction,
    pdp_with_ice,
    shap_bar,
    shap_beeswarm,
    shap_waterfall,
)
from uxplain.protocols import (
    ConformalClassifierProtocol,
    ConformalPredictorProtocol,
    UncertaintyExplainerProtocol,
)
from uxplain.uncertainty.metrics import (
    ClassificationMetric,
    RegressionMetric,
    UncertaintyMetric,
)
from uxplain.uq_explainer import (
    ClassificationExplanationResult,
    ExplanationResult,
    UncertaintyExplanationPipeline,
)

__all__ = [
    "STYLE",
    "CQRConformalPredictor",
    "ClassificationConformalMethod",
    "ClassificationExplanationResult",
    "ClassificationMetric",
    "ConformalClassifierProtocol",
    "ConformalMethod",
    "ConformalPredictorProtocol",
    "CrepesConformalClassifier",
    "CrepesConformalPredictor",
    "ExplanationResult",
    "LIMEExplanation",
    "LimeUncertaintyExplainer",
    "PDPExplanation",
    "PDPUncertaintyExplainer",
    "RegressionMetric",
    "ShapUncertaintyExplainer",
    "UncertaintyExplainerProtocol",
    "UncertaintyExplanationPipeline",
    "UncertaintyMetric",
    "__version__",
    "ice_curves",
    "lime_local",
    "pdp_curve",
    "pdp_importance",
    "pdp_interaction",
    "pdp_with_ice",
    "shap_bar",
    "shap_beeswarm",
    "shap_waterfall",
]