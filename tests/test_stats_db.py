from gateway.stats.db import StatsDb


def _seed(db, ts, base_url="https://a", model="m1", i=10, cr=2, cw=1, o=5, status=200,
          client_protocol=None, upstream_protocol=None, reasoning_effort=None):
    db.insert(ts=ts, base_url=base_url, model=model, input_tokens=i,
              cache_read_tokens=cr, cache_write_tokens=cw, output_tokens=o, status=status,
              client_protocol=client_protocol, upstream_protocol=upstream_protocol,
              reasoning_effort=reasoning_effort)


def test_insert_and_pending(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000)
    _seed(db, 1001, model="m2", i=1, cr=0, cw=0, o=1)
    pending = db.pending()
    assert len(pending) == 2
    assert pending[0]["model"] == "m1"
    assert pending[0]["input_tokens"] == 10
    assert pending[0]["cache_read_tokens"] == 2
    assert pending[0]["cache_write_tokens"] == 1
    assert pending[0]["output_tokens"] == 5


def test_set_cost_clears_pending(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000)
    row = db.pending()[0]
    db.set_cost(row["id"], 1.25)
    assert db.pending() == []
    items, _ = db.list_requests(0, 2000, None, None, 10, 0)
    assert items[0]["cost_usd"] == 1.25


def test_day_minutes_buckets(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    _seed(db, base + 10)                                    # bucket: base
    _seed(db, base + 30, model="m1")                        # bucket: base (same minute)
    _seed(db, base + 90, model="m2", i=7, cr=0, cw=0, o=1)  # bucket: base + 60
    minutes = db.day_minutes(base, base + 120)
    assert [m["ts"] for m in minutes] == [base, base + 60]
    assert minutes[0]["model"] == "m1"
    assert minutes[0]["input"] == 20
    assert minutes[0]["cache_read"] == 4
    assert minutes[1]["model"] == "m2"
    assert minutes[1]["input"] == 7


def test_model_totals_order(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, model="big", i=100, cr=0, cw=0, o=0)
    _seed(db, 1001, model="small", i=10, cr=0, cw=0, o=0)
    totals = db.model_totals(0, 2000)
    assert [t["model"] for t in totals] == ["big", "small"]
    assert totals[0]["tokens"] == 100
    assert totals[0]["calls"] == 1


def test_endpoint_models_hit_rate_and_filter(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, model="m1", i=40, cr=60, cw=0, o=0)
    _seed(db, 1001, model="m1", i=100, cr=0, cw=0, o=0)
    _seed(db, 1002, base_url="https://b", model="m2", i=5, cr=0, cw=0, o=5)
    rows = db.endpoint_models(0, 2000, "https://a")
    assert len(rows) == 1
    assert rows[0]["model"] == "m1"
    assert rows[0]["tokens"] == 200
    assert rows[0]["calls"] == 2
    # hit rate = 60 / (40+60+0 + 100+0+0) = 60/200 = 0.3
    assert rows[0]["cache_hit_rate"] == 0.3
    assert rows[0]["cost"] == 0.0
    assert db.endpoint_models(0, 2000, "https://a", models=["nope"]) == []


def test_list_requests_pagination_and_filters(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    for i in range(5):
        _seed(db, 1000 + i, model=f"m{i % 2}", i=i, cr=0, cw=0, o=0)
    items, total = db.list_requests(0, 2000, None, None, 2, 0)
    assert total == 5
    assert len(items) == 2
    assert items[0]["ts"] == 1004          # desc
    items, _ = db.list_requests(0, 2000, None, None, 2, 2)
    assert items[0]["ts"] == 1002
    items, total = db.list_requests(0, 2000, "https://a", "m0", 10, 0)
    assert total == 3
    items, total = db.list_requests(1002, 1004, None, None, 10, 0)
    assert total == 2                      # [1002, 1004)


def test_list_requests_includes_protocol_and_effort(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, client_protocol="anthropic_messages",
          upstream_protocol="openai_chat", reasoning_effort="high")
    items, _ = db.list_requests(0, 2000, None, None, 10, 0)
    assert items[0]["client_protocol"] == "anthropic_messages"
    assert items[0]["upstream_protocol"] == "openai_chat"
    assert items[0]["reasoning_effort"] == "high"


def test_timeseries_minute_granularity(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, base_url="https://a", model="m1", i=10, cr=2, cw=1, o=5)
    _seed(db, 1000, base_url="https://a", model="m2", i=3, cr=0, cw=0, o=1)
    _seed(db, 1100, base_url="https://a", model="m1", i=5, cr=0, cw=0, o=2)  # different minute
    series = db.timeseries(0, 2000, base_url="https://a", granularity="minute")
    assert len(series) == 3
    assert series[0]["ts"] == 960   # (1000//60)*60
    assert series[0]["model"] == "m1"
    assert series[0]["input"] == 10
    assert series[2]["ts"] == 1080  # (1100//60)*60
    assert series[2]["model"] == "m1"
    assert series[2]["input"] == 5


def test_timeseries_hour_and_day_granularity(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, base_url="https://a", model="m1", i=10)
    _seed(db, 3700, base_url="https://a", model="m1", i=7)  # ~1 hour later
    _seed(db, 90000, base_url="https://a", model="m1", i=3)  # next day
    hour_series = db.timeseries(0, 100000, base_url="https://a", granularity="hour")
    # (1000//3600)*3600 = 0, (3700//3600)*3600 = 3600, (90000//3600)*3600 = 90000... wait 90000//3600=25, *3600 = 90000. Hmm that's also a bucket.
    assert len(hour_series) == 3  # 0-3599 sum=10, 3600-7199 sum=7, 90000-93599 sum=3
    # Actually let's verify: 1000 in bucket 0, 3700 in bucket 3600, 90000 in bucket 90000
    assert hour_series[0]["input"] == 10
    assert hour_series[1]["input"] == 7
    day_series = db.timeseries(0, 100000, base_url="https://a", granularity="day")
    assert len(day_series) == 2  # day 0 and day 86400
    assert day_series[0]["input"] == 17  # 10+7
    assert day_series[1]["input"] == 3


def test_timeseries_no_base_url_returns_all(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, base_url="https://a", model="m1", i=10)
    _seed(db, 1000, base_url="https://b", model="m2", i=5)
    series = db.timeseries(0, 2000, granularity="minute")
    assert len(series) == 2


def test_timeseries_model_filter(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    _seed(db, 1000, base_url="https://a", model="m1", i=10)
    _seed(db, 1000, base_url="https://a", model="m2", i=5)
    series = db.timeseries(0, 2000, base_url="https://a", models=["m1"], granularity="minute")
    assert len(series) == 1
    assert series[0]["model"] == "m1"


def test_metadata_migration_on_existing_db(tmp_path):
    # Old DB without timing/protocol metadata columns gets migrated on open.
    import sqlite3
    p = tmp_path / "old.db"
    conn = sqlite3.connect(p)
    conn.execute("""CREATE TABLE requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL,
        base_url TEXT NOT NULL, model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        cache_write_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        status INTEGER NOT NULL, cost_usd REAL)""")
    conn.commit()
    conn.close()
    db = StatsDb(p)          # opens + migrates
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(requests)")}
    assert {"duration_ms", "ttft_ms", "client_protocol",
            "upstream_protocol", "reasoning_effort"} <= cols
    db.insert(ts=1000, base_url="https://a", model="m1", input_tokens=1,
              cache_read_tokens=0, cache_write_tokens=0, output_tokens=10,
              status=200, duration_ms=500, ttft_ms=200,
              client_protocol="openai_responses",
              upstream_protocol="openai_chat", reasoning_effort="xhigh")
    items, _ = db.list_requests(0, 2000, None, None, 10, 0)
    assert items[0]["client_protocol"] == "openai_responses"
    assert items[0]["upstream_protocol"] == "openai_chat"
    assert items[0]["reasoning_effort"] == "xhigh"
    rows = db.model_tps(0, 2000)
    assert rows == []        # only 1 sample -> omitted
    db.close()


def test_model_tps_samples_and_order(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    # m1: 5 samples; decode tps uses output/(duration - ttft)
    #   (100/(2.0-0.5), 50/(2.0-0.5), 150/(2.0-0.5)) for 3 listed + 2 more
    for out, dur, ttft in ((100, 2000, 500), (50, 2000, 500), (150, 2000, 500),
                           (100, 2000, 500), (100, 2000, 500)):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=out,
                  status=200, duration_ms=dur, ttft_ms=ttft)
    # m2: 5 samples, all decode tps = 200/(1.0-0.5)=400; ttft=500ms
    for out, dur, ttft in ((200, 1000, 500), (200, 1000, 500), (200, 1000, 500),
                           (200, 1000, 500), (200, 1000, 500)):
        db.insert(ts=base, base_url="https://b", model="m2", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=out,
                  status=200, duration_ms=dur, ttft_ms=ttft)
    # m3: status 500 rows don't count
    db.insert(ts=base, base_url="https://c", model="m3", input_tokens=1,
              cache_read_tokens=0, cache_write_tokens=0, output_tokens=100,
              status=500, duration_ms=1000, ttft_ms=100)
    rows = db.model_tps(base, base + 3600)
    assert rows == [
        {"model": "m2", "tps": 400.0, "samples": 5, "avg_ttft_ms": 500.0},
        {"model": "m1", "tps": 66.7, "samples": 5, "avg_ttft_ms": 500.0},
    ]


def test_model_tps_requires_five_samples(tmp_path):
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for out, dur in ((100, 2000), (100, 2000), (100, 2000), (100, 2000)):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=out,
                  status=200, duration_ms=dur)
    assert db.model_tps(base, base + 3600) == []   # only 4 -> omitted


def test_model_tps_ttft_fallback_when_missing(tmp_path):
    # Rows without ttft (non-streaming or legacy): duration <= 3s means the
    # assumed-TTFT offset would leave nothing to divide — x collapses to 0
    # and tps falls back to output/duration.
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=100,
                  status=200, duration_ms=1000)    # no ttft
    rows = db.model_tps(base, base + 3600)
    assert rows == [{"model": "m1", "tps": 100.0, "samples": 5, "avg_ttft_ms": None}]


def test_model_tps_no_ttft_uses_assumed_ttft_offset(tmp_path):
    # Non-streaming rows carry no ttft (unobservable). TPS uses an assumed
    # observed-TTFT of 3s: tps = output / (duration - 3000ms) when duration
    # exceeds 3s.
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=200,
                  status=200, duration_ms=5000)    # no ttft
    rows = db.model_tps(base, base + 3600)
    # 200 / (5.0 - 3.0) = 100
    assert rows == [{"model": "m1", "tps": 100.0, "samples": 5, "avg_ttft_ms": None}]


def test_model_tps_no_ttft_offset_collapses_at_three_seconds(tmp_path):
    # duration exactly 3s: duration - 3000 == 0 -> x collapses to 0.
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=300,
                  status=200, duration_ms=3000)    # no ttft
    rows = db.model_tps(base, base + 3600)
    # 300 / 3.0 = 100 (not a division by zero)
    assert rows == [{"model": "m1", "tps": 100.0, "samples": 5, "avg_ttft_ms": None}]


def test_model_tps_excludes_ttft_outliers(tmp_path):
    # Queueing time IS part of TTFT: even rows with very large TTFT (>10s)
    # are counted in both TPS and avg_ttft (no outlier exclusion).
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    # 5 healthy samples: decode tps = 100/(1.0-0.2) = 125 each; ttft 200ms
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=100,
                  status=200, duration_ms=1000, ttft_ms=200)
    # 2 large-TTFT rows: 12s TTFT is queueing, still counted.
    # Most recent 5 rows are now [outlier, outlier, healthy, healthy, healthy]
    # so the average includes them.
    for _ in range(2):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=100,
                  status=200, duration_ms=20000, ttft_ms=12000)
    rows = db.model_tps(base, base + 3600)
    # 2 outliers: 100/(20-12)=12.5 each; 3 healthy: 125 each
    # tps = (12.5*2 + 125*3)/5 = 80.0; avg_ttft = (12000*2+200*3)/5 = 4920
    assert rows == [{"model": "m1", "tps": 80.0, "samples": 5, "avg_ttft_ms": 4920.0}]


def test_model_tps_short_output_high_ratio_not_outlier(tmp_path):
    # A short-output request where TTFT is 48% of duration is NORMAL
    # (short generations finish quickly) — still counted like any other row.
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=50,
                  status=200, duration_ms=2000, ttft_ms=950)   # 47.5% of dur
    rows = db.model_tps(base, base + 3600)
    assert len(rows) == 1
    assert rows[0]["samples"] == 5
    assert rows[0]["avg_ttft_ms"] == 950.0


def test_model_tps_all_outliers_omits_model(tmp_path):
    # Even when every row has a very large TTFT, it is still counted (no
    # outlier exclusion). 100/(20-15)=20 tokens/s; avg 15000ms.
    db = StatsDb(tmp_path / "usage.db")
    base = 1786320000
    for _ in range(5):
        db.insert(ts=base, base_url="https://a", model="m1", input_tokens=1,
                  cache_read_tokens=0, cache_write_tokens=0, output_tokens=100,
                  status=200, duration_ms=20000, ttft_ms=15000)
    assert db.model_tps(base, base + 3600) == [
        {"model": "m1", "tps": 20.0, "samples": 5, "avg_ttft_ms": 15000.0}
    ]
