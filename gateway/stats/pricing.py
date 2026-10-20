"""Model pricing: resolve user pricing config to effective per-1M-token prices
and compute request costs. Pure functions — no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
import logging

# Defaults when only the input price is configured (Claude-style ratios).
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.1
_OUTPUT_MULT = 5.0

logger = logging.getLogger("llm-gateway")

@dataclass(frozen=True)
class Pricing:
    input: float
    cache_read: float
    cache_write: float
    output: float

    def cost_usd(self, input_tokens: int, cache_read_tokens: int,
                 cache_write_tokens: int, output_tokens: int) -> float:
        return (
            self.input * input_tokens
            + self.cache_read * cache_read_tokens
            + self.cache_write * cache_write_tokens
            + self.output * output_tokens
        ) / 1_000_000.0


def resolve_pricing(raw: dict[str, float] | None) -> Pricing | None:
    """Resolve a user pricing dict to effective prices ($/1M tokens).

    All four fields are optional. Missing cache_write / cache_read / output are
    derived from the input price (×1.25 / ×0.1 / ×5). A missing or empty dict
    returns None (cost 0). Raises ValueError on negative values.
    """
    if not raw:
        return None
    input_price = float(raw.get("input", 0.0))
    if not all(math.isfinite(float(v)) for v in raw.values()) or not math.isfinite(input_price):
        logger.error("pricing values must be finite")
        return None

    if input_price < 0 or any(v < 0 for v in raw.values()):
        logger.error("pricing values must be non-negative")
        return None

    return Pricing(
        input=input_price,
        cache_read=round(float(raw.get("cache_read", input_price * _CACHE_READ_MULT)), 6),
        cache_write=round(float(raw.get("cache_write", input_price * _CACHE_WRITE_MULT)), 6),
        output=round(float(raw.get("output", input_price * _OUTPUT_MULT)), 6),
    )
