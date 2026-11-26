"""03 - Partition counts: shuffle partitions, AQE coalescing, repartition vs coalesce.

Run:   uv run python examples/optimize/03_repartition_coalesce_shuffle_partitions.py

Look for:
  - Too many shuffle partitions = thousands of tiny tasks, each with fixed scheduling overhead.
  - AQE coalescing merges small shuffle partitions after the shuffle, so one setting fits many queries.
  - coalesce(1) has no shuffle, so Spark runs EVERYTHING before it in one task. repartition(1) adds a
    shuffle but keeps the upstream work parallel. Writing one file from heavy work: repartition wins.
"""

import shutil

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark
from pyspark_template.timing import final_plan_text, measure, noop_write, record_benchmark

spark = build_spark("optimize-03-partitions", extra_conf={"spark.sql.adaptive.enabled": "false"})
orders = spark.read.parquet(str(DATA_DIR / "orders"))


def per_customer():
    return orders.groupBy("customer_id").agg(F.sum("quantity").alias("units"), F.count("*").alias("orders"))


results = {}

banner("Shuffle partitions: 2000 (way too many for 5M rows) vs 16")
spark.conf.set("spark.sql.shuffle.partitions", "2000")
measure("2000 shuffle partitions", lambda: noop_write(per_customer()), results)
spark.conf.set("spark.sql.shuffle.partitions", "16")
measure("16 shuffle partitions", lambda: noop_write(per_customer()), results)
record_benchmark(
    "03 shuffle partitions (groupBy 5M rows)",
    "2000 shuffle partitions",
    "16 shuffle partitions",
    results,
    "2000 tiny reduce tasks -> 16 right-sized tasks",
)

banner("AQE: keep 2000, let Spark coalesce after the shuffle")
spark.conf.set("spark.sql.shuffle.partitions", "2000")
spark.conf.set("spark.sql.adaptive.enabled", "true")
measure("2000 + AQE coalesce", lambda: noop_write(per_customer()), results)
aqe_query = per_customer()
aqe_query.collect()
for line in final_plan_text(aqe_query).splitlines():
    if "AQEShuffleRead" in line:
        print("  final plan:", line.strip())
print(f"Partitions actually used after AQE: {aqe_query.rdd.getNumPartitions()}")
record_benchmark(
    "03 AQE partition coalescing",
    "2000 shuffle partitions",
    "2000 + AQE coalesce",
    results,
    "AQEShuffleRead coalesced 2000 partitions down to a few",
)
spark.conf.set("spark.sql.shuffle.partitions", "16")
spark.conf.set("spark.sql.adaptive.enabled", "false")

banner("Writing one output file: coalesce(1) vs repartition(1)")
# Heavy-ish per-row work so the difference is visible.
heavy = orders.withColumn("fingerprint", F.sha2(F.concat_ws("|", "order_id", "customer_id", "status"), 256))
target = OUTPUT_DIR / "optimize03_single_file"


def write_with(df):
    df.write.mode("overwrite").parquet(str(target))


coalesced = heavy.coalesce(1)
repartitioned = heavy.repartition(1)
print(f"Upstream tasks with coalesce(1): {coalesced.rdd.getNumPartitions()} (all work in one task)")
measure("coalesce(1)", lambda: write_with(coalesced), results, runs=2)
measure("repartition(1)", lambda: write_with(repartitioned), results, runs=2)
record_benchmark(
    "03 one output file",
    "coalesce(1)",
    "repartition(1)",
    results,
    "coalesce collapses the whole stage to 1 task; repartition hashes in parallel, then shuffles",
)
shutil.rmtree(target, ignore_errors=True)
print("Rule: coalesce(n) to shrink a little (200 -> 50). repartition(n) when n is tiny or upstream is heavy.")

spark.stop()
