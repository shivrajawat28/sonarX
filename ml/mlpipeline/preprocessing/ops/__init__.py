"""Built-in preprocessing ops. Importing this package registers them all."""
from mlpipeline.preprocessing.ops.normalize import NormalizeIntensityOp
from mlpipeline.preprocessing.ops.contrast import ClaheOp
from mlpipeline.preprocessing.ops.denoise import DenoiseMedianOp, DenoiseNlmOp
from mlpipeline.preprocessing.ops.resize import ResizeLetterboxOp
from mlpipeline.preprocessing.ops.orientation import OptionalFlipWaterfallOp

__all__ = [
    "NormalizeIntensityOp",
    "ClaheOp",
    "DenoiseMedianOp",
    "DenoiseNlmOp",
    "ResizeLetterboxOp",
    "OptionalFlipWaterfallOp",
]
