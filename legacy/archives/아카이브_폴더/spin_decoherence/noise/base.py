"""
Base classes and exceptions for noise models.

This module provides common exception classes used across all noise models.
"""


class NumericalStabilityError(Exception):
    """수치적 불안정성 에러 (Numerical Stability Error)"""
    pass


class InvalidParameterError(ValueError):
    """유효하지 않은 파라미터 에러 (Invalid Parameter Error)"""
    pass

