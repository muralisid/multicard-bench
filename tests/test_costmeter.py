"""Tests for cost accounting.

The budget cap is a safety control, so it is tested like one: it must refuse the
call that would cross the line, not report the overspend afterwards.
"""

import pytest

from multicard.llm.costmeter import (BudgetExceeded, CostMeter,
                                     PRICES_USD_PER_MTOK)


def test_cost_is_tokens_times_published_price():
    m = CostMeter(max_usd=100)
    m.record("generative-cheap", 1_000_000, 1_000_000)
    p = PRICES_USD_PER_MTOK["generative-cheap"]
    assert m.total_usd() == pytest.approx(p["in"] + p["out"])


def test_local_encoding_is_free():
    m = CostMeter(max_usd=0.0)
    m.record("encoder-local", 50_000_000, 0)   # must not raise
    assert m.total_usd() == 0.0


def test_budget_cap_raises_on_the_offending_call():
    m = CostMeter(max_usd=0.001)
    with pytest.raises(BudgetExceeded):
        m.record("generative-frontier", 10_000_000, 10_000_000)


def test_unknown_tier_is_rejected_rather_than_priced_at_zero():
    with pytest.raises(KeyError):
        CostMeter().record("mystery-model", 100, 100)


def test_report_breaks_down_by_tier_and_dates_the_prices():
    m = CostMeter(max_usd=100)
    m.record("generative-cheap", 1000, 100)
    m.record("generative-frontier", 2000, 200)
    d = m.as_dict()
    assert set(d["by_tier"]) == {"generative-cheap", "generative-frontier"}
    assert d["total_calls"] == 2
    assert d["prices_as_of"]
    assert d["by_tier"]["generative-frontier"]["usd"] > d["by_tier"]["generative-cheap"]["usd"]
