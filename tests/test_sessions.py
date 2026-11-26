from datetime import datetime

from pyspark.sql import Row
from pyspark.testing import assertDataFrameEqual

from pyspark_template.transforms.sessions import sessionize_clicks, summarize_sessions


def make_clicks(spark):
    return spark.createDataFrame(
        [
            ("e1", 1, datetime(2026, 3, 1, 10, 0)),
            ("e2", 1, datetime(2026, 3, 1, 10, 20)),  # 20 min gap: same session
            ("e3", 1, datetime(2026, 3, 1, 11, 0)),  # 40 min gap: new session
            ("e4", 2, datetime(2026, 3, 1, 10, 0)),  # other user: own session
        ],
        "event_id string, customer_id long, event_ts timestamp",
    )


def test_sessionize_starts_new_session_after_gap(spark):
    result = sessionize_clicks(make_clicks(spark), gap_minutes=30).select("event_id", "session_id")
    expected = [
        Row(event_id="e1", session_id="1-1"),
        Row(event_id="e2", session_id="1-1"),
        Row(event_id="e3", session_id="1-2"),
        Row(event_id="e4", session_id="2-1"),
    ]
    assertDataFrameEqual(result, expected)


def test_summarize_sessions(spark):
    sessions = summarize_sessions(sessionize_clicks(make_clicks(spark), gap_minutes=30))
    result = sessions.select("session_id", "event_count", "duration_seconds")
    expected = [
        Row(session_id="1-1", event_count=2, duration_seconds=1200),
        Row(session_id="1-2", event_count=1, duration_seconds=0),
        Row(session_id="2-1", event_count=1, duration_seconds=0),
    ]
    assertDataFrameEqual(result, expected)
