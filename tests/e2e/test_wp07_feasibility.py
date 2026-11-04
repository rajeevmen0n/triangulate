"""WP07 go/no-go gate: adsb.lol contains props explaining both Oyster Bay demo events."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.network, pytest.mark.slow]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "wp07_feasibility.py"


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("wp07_feasibility", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()


def test_gate_go(result):
    e1, e2 = result["events"][1], result["events"][2]
    # each event is explained by a non-jet track under the plan's acceptance criteria
    assert e1["plan_criteria"], "no event-1 candidate"
    assert e2["plan_criteria"], "no event-2 candidate"
    # a pair of distinct aircraft 921 +- 10 s apart in heard time
    assert result["go"]
    # and it holds for the video's own measured speed/distance ranges
    assert result["pairs_within_10s_of_921"]
    for p in e1["strict"] + e2["strict"]:
        assert p["keep"] and p["coverage"]["max_gap_s"] <= 30.0
        assert abs(p["solve_emission_err_s"]) < 0.05
