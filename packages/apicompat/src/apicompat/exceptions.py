"""Conversion exceptions."""

from typing import Any, Dict, Optional


class ConversionError(Exception):
    """Base exception for conversion errors."""

    def __init__(
        self,
        message: str,
        source_protocol: Optional[str] = None,
        target_protocol: Optional[str] = None,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.source_protocol = source_protocol
        self.target_protocol = target_protocol
        self.field = field
        self.details = details or {}


class CapabilityNotSupportedError(ConversionError):
    """
    Raised when the target protocol doesn't support a required capability.
    """

    def __init__(
        self,
        capability: str,
        source_protocol: Optional[str] = None,
        target_protocol: Optional[str] = None,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"Capability '{capability}' is not supported by target protocol"
        if suggestion:
            message += f". {suggestion}"
        super().__init__(
            message=message,
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            field=capability,
            details=details or {},
        )
        self.capability = capability
        self.suggestion = suggestion


class ValidationError(ConversionError):
    """
    Raised when input validation fails.

    Examples:
    - Missing required field (e.g., max_tokens for Anthropic)
    - Invalid value range (e.g., temperature > 1 for Anthropic)
    - Invalid content structure
    """

    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None,
        expected: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        full_message = f"Validation error for '{field}': {message}"
        super().__init__(
            message=full_message,
            field=field,
            details=details or {},
        )
        self.value = value
        self.expected = expected
