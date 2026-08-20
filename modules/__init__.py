"""Reusable components for the benchmark.

Submodules are intentionally not imported here. This keeps optional model and
representation dependencies (for example PyTorch, RDKit, and MACE) from being
required when a caller uses only a scikit-learn baseline.
"""

__all__: list[str] = []
