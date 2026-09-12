"""Filter rules (Section 9.3). Importing registers all rules in the rule registry."""
from mlpipeline.filtering.rules.min_size import MinSizeRule
from mlpipeline.filtering.rules.aspect_ratio import AspectRatioRule
from mlpipeline.filtering.rules.edge_clip import EdgeClipRule
from mlpipeline.filtering.rules.intensity_outlier import IntensityOutlierRule
from mlpipeline.filtering.rules.shadow_ratio import ShadowRatioRule  # experimental
from mlpipeline.filtering.rules.class_rules import ClassSpecificRule
from mlpipeline.filtering.rules.validator_stub import SecondaryValidatorStub

__all__ = [
    "MinSizeRule",
    "AspectRatioRule",
    "EdgeClipRule",
    "IntensityOutlierRule",
    "ShadowRatioRule",
    "ClassSpecificRule",
    "SecondaryValidatorStub",
]
