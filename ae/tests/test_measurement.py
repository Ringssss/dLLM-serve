import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import summarize, schedule


def test_drain_and_failures_are_accounted_separately():
    rows = [dict(ok=True, finished_s=2, latency_s=2, sent_s=0, scheduled_s=0),
            dict(ok=True, finished_s=15, latency_s=14, sent_s=1.2, scheduled_s=1),
            dict(ok=False, finished_s=20, latency_s=18, sent_s=2, scheduled_s=2)]
    s = summarize(rows, 10)
    assert s['submitted'] == 3 and s['failed'] == 1
    assert s['measurement_completions_per_s'] == .1
    assert s['drained_requests_per_s'] == .1
    assert s['p90_s'] > 10  # drain latency is included, not clipped at window end


def test_pair_schedules_are_identical_for_a_seed():
    a = schedule(2, 10, 42)
    assert a == schedule(2, 10, 42)
    assert a != schedule(2, 10, 43)
    assert a == sorted(a) and all(0 <= t < 10 for t in a)
