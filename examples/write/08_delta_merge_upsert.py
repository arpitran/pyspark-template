"""08 - Delta Lake: MERGE (upsert), time travel, and schema enforcement.

Run:   uv run python examples/write/08_delta_merge_upsert.py
(The first run downloads the Delta jars, so it is slower.)

Look for:
  - MERGE updates matched rows and inserts new ones in one atomic commit.
  - The source must have one row per key. Duplicates in the source make MERGE fail; dedupe first.
  - Each write is a new table version. versionAsOf reads an older one.
  - Delta rejects a write with an unexpected column unless you opt in with mergeSchema.
"""

import shutil

from delta.tables import DeltaTable
from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark, short_error
from pyspark_template.transforms.windows import latest_order_per_customer

spark = build_spark("08-delta", with_delta=True)
table_path = str(OUTPUT_DIR / "08_customers_delta")
shutil.rmtree(table_path, ignore_errors=True)

customers = spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(str(DATA_DIR / "customers.csv"))

banner("Version 0: create the Delta table from the first 1,000 customers")
customers.filter("customer_id <= 1000").write.format("delta").save(table_path)
print(f"Rows: {spark.read.format('delta').load(table_path).count():,}")

banner("Build a batch of changes: 3 tier upgrades + 2 new customers")
changes = spark.createDataFrame(
    [
        (1, "Customer 1", "customer1@example.com", "US", None, "gold"),
        (2, "Customer 2", "customer2@example.com", "CA", None, "gold"),
        (3, "Customer 3", "customer3-new@example.com", "GB", None, "gold"),
        (100001, "New Customer A", "new.a@example.com", "IN", None, "free"),
        (100002, "New Customer B", "new.b@example.com", "DE", None, "free"),
    ],
    schemas.CUSTOMERS,
).withColumn("signup_date", F.current_date())

banner("MERGE: update matched, insert not matched")
target = DeltaTable.forPath(spark, table_path)
(
    target.alias("t")
    .merge(changes.alias("s"), "t.customer_id = s.customer_id")
    .whenMatchedUpdate(set={"tier": "s.tier", "email": "s.email"})
    .whenNotMatchedInsertAll()
    .execute()
)
spark.read.format("delta").load(table_path).filter("customer_id IN (1, 2, 3, 100001, 100002)").orderBy(
    "customer_id"
).show()
target.history().select(
    "version", "operation", "operationMetrics.numTargetRowsUpdated", "operationMetrics.numTargetRowsInserted"
).show(truncate=False)

banner("Duplicate keys in the source make MERGE fail")
duplicated = changes.union(changes.filter("customer_id = 1"))
try:
    target.alias("t").merge(
        duplicated.alias("s"), "t.customer_id = s.customer_id"
    ).whenMatchedUpdateAll().execute()
except Exception as error:
    print("MERGE raised:", short_error(error))
print("Fix: dedupe the source first, keeping the latest row per key (row_number window, see write/05).")

banner("Real pattern: upsert each customer's latest order into a 'last_order' table")
orders = spark.read.parquet(str(DATA_DIR / "orders")).filter("customer_id <= 1000")
latest = latest_order_per_customer(orders).select("customer_id", "order_id", "order_ts")
last_order_path = str(OUTPUT_DIR / "08_last_order_delta")
shutil.rmtree(last_order_path, ignore_errors=True)
latest.filter("order_ts < '2026-02-01'").write.format("delta").save(last_order_path)
last_order = DeltaTable.forPath(spark, last_order_path)
(
    last_order.alias("t")
    .merge(latest.alias("s"), "t.customer_id = s.customer_id")
    .whenMatchedUpdateAll(condition="s.order_ts > t.order_ts")
    .whenNotMatchedInsertAll()
    .execute()
)
last_order.history().select(
    "version", "operation", "operationMetrics.numTargetRowsUpdated", "operationMetrics.numTargetRowsInserted"
).show(truncate=False)

banner("Time travel: read version 0 of the customers table")
version_zero = spark.read.format("delta").option("versionAsOf", 0).load(table_path)
print(
    f"Version 0 rows: {version_zero.count():,}   latest rows: {spark.read.format('delta').load(table_path).count():,}"
)

banner("Schema enforcement: a write with an extra column is rejected")
with_extra_column = changes.withColumn("loyalty_points", F.lit(10))
try:
    with_extra_column.write.format("delta").mode("append").save(table_path)
except Exception as error:
    print("Append raised:", short_error(error))
with_extra_column.write.format("delta").mode("append").option("mergeSchema", "true").save(table_path)
print("With mergeSchema=true the column is added:", spark.read.format("delta").load(table_path).columns)

spark.stop()
