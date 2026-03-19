"""
I-FailSense: VLM-based binary failure classifier for robot manipulation.

Public API:
    from i_failsense.model import FailSense, process_input, train_model
    from i_failsense.inference import batch_inference
    from i_failsense.load_dataset import load_data, augment_droid_dataset
    from i_failsense.visualization import visualization_report
"""

from .model import FailSense, process_input, train_model
from .inference import batch_inference
from .load_dataset import load_data, augment_droid_dataset
from .visualization import visualization_report

__all__ = [
    "FailSense",
    "process_input",
    "train_model",
    "batch_inference",
    "load_data",
    "augment_droid_dataset",
    "visualization_report",
]
