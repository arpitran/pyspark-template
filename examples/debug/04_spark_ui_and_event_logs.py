"""04 - The Spark UI, its REST API, and event logs: find the slow stage and the slow task.

Run:   uv run python examples/debug/04_spark_ui_and_event_logs.py
       KEEP_UI=1 uv run python examples/debug/04_spark_ui_and_event_logs.py   (keeps the UI open to browse)

Look for:
  - setJobDescription labels jobs, so you can find them in the UI.
  - The same numbers the UI shows are in a REST API at /api/v1. Scripts can check them.
  - Task skew: in a skewed stage the slowest task takes far longer than the median task.
    The UI shows this in the stage page's "Summary Metrics" (Median vs Max).
  - The event log lets the History Server replay the UI after the app has finished.
"""

import json
import os
import statistics
import urllib.request

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, EVENT_LOG_DIR, banner, build_spark

# AQE off for this demo so the skew stays visible. optimize/02 shows AQE fixing it.
spark = build_spark(
    "debug-04-spark-ui",
    event_log=True,
    extra_conf={
        "spark.sql.adaptive.enabled": "false",
        "spark.sql.autoBroadcastJoinThreshold": "-1",
        "spark.sql.shuffle.partitions": "32",
    },
)
context = spark.sparkContext
ui_url = context.uiWebUrl
print(f"Spark UI: {ui_url}")


def get_json(path):
    with urllib.request.urlopen(f"{ui_url}/api/v1{path}") as response:
        return json.loads(response.read())


orders = spark.read.parquet(str(DATA_DIR / "orders"))
customers = (
    spark.range(1, 100_001)
    .withColumnRenamed("id", "customer_id")
    .withColumn("segment", F.col("customer_id") % 7)
)

banner("Run a skewed join (customer 1 has ~30% of orders) with a job description")
context.setJobDescription("skewed join: orders x customers by customer_id")
orders.join(customers, "customer_id").groupBy("segment").count().write.format("noop").mode("overwrite").save()
context.setJobDescription(None)

banner("Ask the REST API what happened")
app_id = context.applicationId
stages = get_json(f"/applications/{app_id}/stages?status=complete")
print(f"{'stage':>5} {'tasks':>5} {'seconds':>8} {'shuffle read rows':>18}  name")
join_stage = None
for stage in stages:
    seconds = stage["executorRunTime"] / 1000
    shuffle_rows = stage.get("shuffleReadRecords", 0)
    print(f"{stage['stageId']:>5} {stage['numTasks']:>5} {seconds:>8.2f} {shuffle_rows:>18,}  {stage['name'][:50]}")
    # The join runs in the stage that reads the most shuffled rows.
    if join_stage is None or shuffle_rows > join_stage.get("shuffleReadRecords", 0):
        join_stage = stage
slowest = join_stage

banner(f"Tasks in the join stage ({slowest['stageId']}): one per shuffle partition")
tasks = get_json(
    f"/applications/{app_id}/stages/{slowest['stageId']}/{slowest['attemptId']}/taskList?length=1000"
)
durations = []
records = []
for task in tasks:
    durations.append(task["duration"])
    shuffle_read = task.get("taskMetrics", {}).get("shuffleReadMetrics", {}).get("recordsRead", 0)
    records.append(shuffle_read)
median_ms = statistics.median(durations)
max_ms = max(durations)
print(
    f"tasks: {len(durations)}   median: {median_ms} ms   max: {max_ms} ms   max/median: {max_ms / max(median_ms, 1):.1f}x"
)
print(f"shuffle records read - median: {statistics.median(records):,.0f}   max: {max(records):,}")
records_ratio = max(records) / max(statistics.median(records), 1)
print(f"records max/median: {records_ratio:.1f}x")
if records_ratio > 3:
    print("Skew: one task reads many times more rows than a typical task. That is customer 1's partition.")
    print("On a laptop the time gap looks small. On a cluster with 100x the data, that one task decides")
    print("when the stage ends, and it is the one that spills or runs out of memory. See optimize/02.")
else:
    print("No strong skew in this stage.")

banner("Event log for the History Server")
# Spark 4 writes each event log as a folder (eventlog_v2_<app id>) of rolling files.
current_log = EVENT_LOG_DIR / f"eventlog_v2_{app_id}"
print(f"This run's event log: spark-events/{current_log.name}/ (written fully when the app stops)")
print("Replay later with the Spark History Server:")
print(
    "  spark.history.fs.logDirectory=file://<repo>/spark-events  then  $SPARK_HOME/sbin/start-history-server.sh"
)
print("  and open http://localhost:18080")

if os.environ.get("KEEP_UI") == "1":
    input(f"\nUI is live at {ui_url}. Press Enter to stop... ")

spark.stop()
