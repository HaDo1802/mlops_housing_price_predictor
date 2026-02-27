"""Compatibility shim for old pickle import paths.

Legacy artifacts referenced `src.features_engineer.preprocessor.ProductionPreprocessor`.
This module preserves that path and re-exports the current implementation.
"""

from housing_predictor.features.preprocessor import ProductionPreprocessor

__all__ = ["ProductionPreprocessor"]

