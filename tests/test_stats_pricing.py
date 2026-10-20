from gateway.stats.pricing import Pricing, resolve_pricing


def test_only_input_derives_defaults():
    p = resolve_pricing({"input": 3.0})
    assert p is not None
    assert p.input == 3.0
    assert p.cache_read == 0.3          # input × 0.1
    assert p.cache_write == 3.75        # input × 1.25
    assert p.output == 15.0             # input × 5


def test_all_four_explicit():
    p = resolve_pricing({"input": 1, "cache_read": 0.2, "cache_write": 2, "output": 5})
    assert p == Pricing(input=1.0, cache_read=0.2, cache_write=2.0, output=5.0)


def test_partial_without_input_defaults_missing_to_zero():
    p = resolve_pricing({"output": 10})
    assert p is not None
    assert p.input == 0.0
    assert p.cache_read == 0.0
    assert p.cache_write == 0.0
    assert p.output == 10.0


def test_empty_and_none_return_none():
    assert resolve_pricing(None) is None
    assert resolve_pricing({}) is None


def test_negative_values_rejected():
    assert resolve_pricing({"input": -1}) is None
    assert resolve_pricing({"input": 1, "output": -2}) is None


def test_cost_calculation():
    p = Pricing(input=3.0, cache_read=0.3, cache_write=3.75, output=15.0)
    cost = p.cost_usd(
        input_tokens=1_000_000, cache_read_tokens=500_000,
        cache_write_tokens=200_000, output_tokens=100_000,
    )
    # 3.0 + 0.15 + 0.75 + 1.5 = 5.4
    assert cost == 5.4
