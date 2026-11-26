"""Sessionization: group a user's clicks into visits.

Rule: a new session starts when the gap since the user's previous click is
more than `gap_minutes`, or when it is the user's first click.

The trick, in three steps, all with window functions (no UDF, no groupBy):
  1. lag() to get the previous click time
  2. flag = 1 when this click starts a new session, else 0
  3. running sum of the flag = session number (1, 1, 1, 2, 2, 3 ...)
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def sessionize_clicks(clicks: DataFrame, gap_minutes=30) -> DataFrame:
    in_time_order = Window.partitionBy("customer_id").orderBy("event_ts", "event_id")

    with_previous = clicks.withColumn("previous_ts", F.lag("event_ts").over(in_time_order))

    seconds_since_previous = F.col("event_ts").cast("long") - F.col("previous_ts").cast("long")
    is_new_session = F.col("previous_ts").isNull() | (seconds_since_previous > gap_minutes * 60)
    flagged = with_previous.withColumn("new_session_flag", F.when(is_new_session, 1).otherwise(0))

    so_far = in_time_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    numbered = flagged.withColumn("session_number", F.sum("new_session_flag").over(so_far))

    session_id = F.concat_ws("-", F.col("customer_id").cast("string"), F.col("session_number").cast("string"))
    return numbered.withColumn("session_id", session_id).drop("previous_ts", "new_session_flag")


def summarize_sessions(sessionized: DataFrame) -> DataFrame:
    """One row per session: start, end, length in seconds, number of events."""
    return sessionized.groupBy("customer_id", "session_id").agg(
        F.min("event_ts").alias("session_start"),
        F.max("event_ts").alias("session_end"),
        F.count("*").alias("event_count"),
        (F.max("event_ts").cast("long") - F.min("event_ts").cast("long")).alias("duration_seconds"),
    )
