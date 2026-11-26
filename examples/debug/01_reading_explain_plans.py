"""01 - Reading query plans. This is the main debugging tool for slow or wrong Spark code.

Run:   uv run python examples/debug/01_reading_explain_plans.py

Read a physical plan from the bottom up: data is scanned at the bottom and flows up to the result.

Words to look for:
  FileScan ... PartitionFilters   folders skipped by partition pruning (cheap: files never opened)
  FileScan ... PushedFilters      filters handed to the Parquet reader (row groups skipped using min/max stats)
  Exchange                        a SHUFFLE: data moves between tasks over the network/disk. Usually the cost.
  HashAggregate (partial, final)  aggregate inside each task first, shuffle the small result, then finish
  BroadcastHashJoin               small side copied to every task, no shuffle of the big side
  SortMergeJoin                   both sides shuffled and sorted on the key
  AdaptiveSparkPlan isFinalPlan   AQE re-plans at runtime. The final plan is only known after an action.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import final_plan_text

spark = build_spark("debug-01-explain")

orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))

query = (
    orders.filter(F.col("order_date") >= "2026-03-01")  # partition column -> PartitionFilters
    .filter(F.col("quantity") >= 3)  # data column -> PushedFilters
    .join(products, "product_id")
    .groupBy("category")
    .agg(F.sum(F.col("quantity") * F.col("unit_price")).alias("revenue"))
)

banner("explain() - physical plan only")
query.explain()

banner("explain('extended') - parsed -> analyzed -> optimized logical -> physical")
print("The optimized logical plan shows what Catalyst rewrote: filters pushed below the join,")
print("unused columns pruned, constant expressions folded.")
query.explain("extended")

banner("explain('formatted') - plan outline plus a details block per operator (easiest to read)")
query.explain("formatted")

banner("explain('cost') - optimized logical plan with size estimates (Statistics(sizeInBytes=...))")
query.explain("cost")
print("The size estimate decides whether a join side is under spark.sql.autoBroadcastJoinThreshold.")

banner("Before vs after running: AQE changes the plan at runtime")
print("Before an action the root says isFinalPlan=false. Run it, then read the executed plan:")
# collect() runs THIS DataFrame's plan. A write would build a new plan and leave this one unexecuted.
query.collect()
final_plan = final_plan_text(query)
print(final_plan)
print("Look for AQEShuffleRead (coalesced partitions) and whether the join stayed a BroadcastHashJoin.")

banner("Checklist when a query is slow")
checklist = [
    "1. Count the Exchange nodes. Each one is a shuffle. Can a broadcast or a pre-partitioned input remove one?",
    "2. Does the FileScan show PartitionFilters / PushedFilters? If not, a function on the column blocked them.",
    "3. Is there a SortMergeJoin with one small side? It should be a BroadcastHashJoin.",
    "4. Is a Python UDF (BatchEvalPython / ArrowEvalPython) in the hot path? Use a built-in function.",
    "5. Open the Spark UI (debug/04): is one task much slower than the rest? That is skew (optimize/02).",
]
for line in checklist:
    print(line)

spark.stop()
