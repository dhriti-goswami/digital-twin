"""Explainable physics-guided digital twin for T1D glucose forecasting.

This package is the single source of truth for the research pipeline. Nothing in
``src/`` or ``scripts/`` is imported here: the legacy tree is retained only for
the deployed application and is not used to produce any reported number.

Layout
------
``twin.physio``   Mechanistic models (Bergman minimal model, subcutaneous
                  insulin absorption, gut carbohydrate absorption).
``twin.metrics``  Accuracy, clinical, error-grid, and statistical reporting.
``twin.data``     Dataset parsers, gap-aware sequencing, feature contract,
                  split protocols.
``twin.models``   Forecasting models and training loops.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
