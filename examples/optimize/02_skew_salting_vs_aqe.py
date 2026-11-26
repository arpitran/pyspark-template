"""02 - Data skew: one hot key (customer 1, ~30% of orders) makes one task do most of the work.

Run:   uv run python examples/optimize/02_skew_salting_vs_aqe.py

Three runs of the same join, orders x customers on customer_id (broadcast disabled so it must shuffle):
  BEFORE   plain sort-merge join, AQE off. The task holding customer 1 reads ~1.5M rows; the median task ~0.1M.
  AFTER A  AQE skew-join handling: Spark spots the oversized partition at runtime and splits it.
           Look for "SortMergeJoin(skew=true)" in the final plan.
  AFTER B  Manual salting: spread the hot key over N sub-keys. Works on any engine/version,
           costs N copies of the matching rows on the other side.

The local data is small, so the AQE skew thresholds are lowered to match it. Defaults are
skewedPartitionFactor=5 and skewedPartitionThresholdInBytes=256MB, tuned for real clusters.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import final_plan_text, measure, record_benchmark

SALT_BUCKETS = 16

spark = build_spark(
    "optimize-02-skew",
    extra_conf={
        "spark.sql.autoBroadcastJoinThreshold": "-1",
        "spark.sql.shuffle.partitions": "32",
    },
)
orders = spark.read.parquet(str(DATA_DIR / "orders"))
customers = spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(str(DATA_DIR / "customers.csv"))
customers = customers.select("customer_id", "country", "tier")


def revenue_rows_by_country(joined):
    return joined.groupBy("country", "tier").agg(
        F.count("*").alias("orders"), F.sum("quantity").alias("units")
    )


banner("How skewed is the key?")
orders.groupBy("customer_id").count().orderBy(F.col("count").desc()).show(3)

results = {}

banner("BEFORE: AQE off, plain sort-merge join")
spark.conf.set("spark.sql.adaptive.enabled", "false")
before = revenue_rows_by_country(orders.join(customers, "customer_id"))
# Build a fresh DataFrame inside every timed run. Collecting the SAME DataFrame twice reuses
# its shuffle files, so the second run would skip the join entirely.
measure(
    "plain join", lambda: revenue_rows_by_country(orders.join(customers, "customer_id")).collect(), results
)

banner("AFTER A: AQE skew join")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "3")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "4MB")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "4MB")
after_aqe = revenue_rows_by_country(orders.join(customers, "customer_id"))
measure(
    "AQE skew join", lambda: revenue_rows_by_country(orders.join(customers, "customer_id")).collect(), results
)
after_aqe.collect()
aqe_plan = final_plan_text(after_aqe)
print("Final plan has a skew-split join:", "skew=true" in aqe_plan)
for line in aqe_plan.splitlines():
    if "SortMergeJoin" in line or "AQEShuffleRead" in line:
        print("  ", line.strip()[:120])

banner("AFTER B: manual salting, AQE off")
spark.conf.set("spark.sql.adaptive.enabled", "false")
# Orders: each row gets a random salt 0..N-1, so customer 1's rows spread over N partitions.
salted_orders = orders.withColumn("salt", (F.rand(seed=7) * SALT_BUCKETS).cast("int"))
# Customers: each row is copied N times, once per salt value, so every salted order still finds its match.
salt_values = []
for salt in range(SALT_BUCKETS):
    salt_values.append(F.lit(salt))
salted_customers = customers.withColumn("salt", F.explode(F.array(*salt_values)))
salted_join = salted_orders.join(salted_customers, ["customer_id", "salt"]).drop("salt")
after_salt = revenue_rows_by_country(salted_join)
measure("salted join", lambda: revenue_rows_by_country(salted_join).collect(), results)

banner("Same answer all three ways?")
before_rows = sorted(before.collect())
print("AQE result matches:", before_rows == sorted(after_aqe.collect()))
print("Salted result matches:", before_rows == sorted(after_salt.collect()))
print(f"Salting cost: customers grew from {customers.count():,} to {salted_customers.count():,} rows.")
print("Better salting: copy only the hot keys, or salt every key but with fewer buckets.")

record_benchmark(
    "02 skew: AQE skew join",
    "plain join",
    "AQE skew join",
    results,
    "hot partition split by AQE (SortMergeJoin skew=true)",
)
record_benchmark(
    "02 skew: manual salting",
    "plain join",
    "salted join",
    results,
    f"customer 1 spread over {SALT_BUCKETS} partitions; no gain at this size: customers copied 16x (1.6M rows)",
)

spark.stop()
