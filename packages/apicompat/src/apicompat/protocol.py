"""
Supported API protocols.
"""

from enum import Enum


class Protocol(Enum):
    """Supported API protocols."""
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
