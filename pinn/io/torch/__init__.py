# pinn/io/torch/__init__.py
"""
Torch-specific IO components for PiNN.

This subpackage contains implementations that are only needed when the
PyTorch backend is active (e.g., caching, dataset wrappers, collation,
preprocessing). Keep this module import-light: avoid eager imports here
to prevent circular dependencies and unnecessary backend coupling.
"""
