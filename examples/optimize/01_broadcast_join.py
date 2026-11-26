"""01 - Broadcast join: stop shuffling 5M rows to join a 1,000-row table.

Run:   uv run python examples/optimize/01_broadcast_join.py

Before: SortMergeJoin. Both sides are shuffled by product_id and sorted, including all 5M orders.
After:  BroadcastHashJoin. The 1,000 products are copied to every task; orders never move.

Look for:
  - "Exchange hashpartitioning(product_id" under the join in the BEFORE plan, gone in the AFTER plan.
  - Spark broadcasts automatically when it knows a side is under spark.sql.autoBroadcastJoinThreshold
    (10MB). It often does NOT know: sizes of filtered/aggregated/UDF outputs are guesses. F.broadcast() is the hint.
  - Do not broadcast big tables: every task holds a full copy, and the driver builds it first.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import measure, noop_write, plan_contains, record_benchmark

spark = build_spark("optimize-01-broadcast")
orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))


def revenue_by_category(product_side):
    joined = orders.join(product_side, "product_id")
    return joined.groupBy("category").agg(F.sum(F.col("quantity") * F.col("unit_price")).alias("revenue"))


results = {}

banner("BEFORE: automatic broadcast and AQE turned off -> SortMergeJoin")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
spark.conf.set("spark.sql.adaptive.enabled", "false")
before = revenue_by_category(products)
before.explain()
print("SortMergeJoin in plan:", plan_contains(before, "SortMergeJoin"))
measure("sort-merge join", lambda: noop_write(before), results)

banner("AFTER: same settings, explicit broadcast hint -> BroadcastHashJoin")
after = revenue_by_category(F.broadcast(products))
after.explain()
print("BroadcastHashJoin in plan:", plan_contains(after, "BroadcastHashJoin"))
measure("broadcast join", lambda: noop_write(after), results)

record_benchmark(
    "01 broadcast join (5M orders x 1K products)",
    "sort-merge join",
    "broadcast join",
    results,
    "SortMergeJoin + 2 Exchanges -> BroadcastHashJoin, orders not shuffled",
)

banner("Default settings: AQE + 10MB threshold pick broadcast on their own here")
spark.conf.unset("spark.sql.autoBroadcastJoinThreshold")
spark.conf.unset("spark.sql.adaptive.enabled")
default_plan = revenue_by_category(products)
default_plan.collect()
print(
    "BroadcastHashJoin in final plan:", plan_contains(default_plan, "BroadcastHashJoin", use_final_plan=True)
)
print("Because the CSV is tiny and Spark can measure it. Keep the hint for sides whose size Spark can't see.")

spark.stop()
