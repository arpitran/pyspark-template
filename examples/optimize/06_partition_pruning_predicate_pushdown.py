"""06 - Read less data: partition pruning, predicate pushdown, column pruning.

Run:   uv run python examples/optimize/06_partition_pruning_predicate_pushdown.py

Look for, in the FileScan line of each plan:
  PartitionFilters: [...]  -> folders Spark skips without opening a file
  PushedFilters: [...]     -> filters the Parquet reader applies using row-group min/max statistics
  ReadSchema: struct<...>  -> the only columns read from disk (Parquet is columnar)

The common mistake: filtering on a derived value (to_date(order_ts)) instead of the partition
column (order_date). Same rows, but Spark must open every file to check.
"""

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import measure, plan_text, record_benchmark

spark = build_spark("optimize-06-pruning")
orders = spark.read.parquet(str(DATA_DIR / "orders"))


def scan_line(df):
    """Pull the interesting parts of the FileScan node out of the formatted plan."""
    keep = []
    for line in plan_text(df).splitlines():
        stripped = line.strip()
        if stripped.startswith(("PartitionFilters", "PushedFilters", "ReadSchema")):
            keep.append("  " + stripped[:150])
    return "\n".join(keep)


results = {}

banner("BEFORE: filter on to_date(order_ts) - same day, but no partition pruning")
before = orders.filter(F.to_date("order_ts") == "2026-03-01").agg(F.sum("quantity"))
print(scan_line(before))
# A fresh DataFrame per run: collecting the same DataFrame again can reuse earlier work.
measure(
    "filter on to_date(order_ts)",
    lambda: orders.filter(F.to_date("order_ts") == "2026-03-01").agg(F.sum("quantity")).collect(),
    results,
)

banner("AFTER: filter on the partition column order_date")
after = orders.filter(F.col("order_date") == "2026-03-01").agg(F.sum("quantity"))
print(scan_line(after))
measure(
    "filter on order_date (partition)",
    lambda: orders.filter(F.col("order_date") == "2026-03-01").agg(F.sum("quantity")).collect(),
    results,
)
print("Same answer:", before.collect() == after.collect())
record_benchmark(
    "06 partition pruning (1 of 90 days)",
    "filter on to_date(order_ts)",
    "filter on order_date (partition)",
    results,
    "PartitionFilters [] -> [order_date = 2026-03-01]: 1 folder read instead of 90",
)

banner("Predicate pushdown: a plain comparison is pushed, a cast column is not")
pushed = orders.filter(F.col("quantity") == 5)
not_pushed = orders.filter(F.col("quantity").cast("string") == "5")
print("quantity == 5:")
print(scan_line(pushed))
print("cast(quantity as string) == '5':")
print(scan_line(not_pushed))
print("Pushdown pays off when values are clustered (sorted data, ids, dates). On random data like this,")
print("every row group holds every value, so min/max can't skip anything; that is why sort order matters.")

banner("Column pruning: select what you need before the action")
all_columns = orders.filter(F.col("order_date") >= "2026-03-01")
two_columns = all_columns.select("customer_id", "quantity")
print("All columns:")
print(scan_line(all_columns))
print("Two columns:")
print(scan_line(two_columns))
measure("read all columns", lambda: all_columns.write.format("noop").mode("overwrite").save(), results)
measure("read 2 columns", lambda: two_columns.write.format("noop").mode("overwrite").save(), results)
record_benchmark(
    "06 column pruning (Parquet)",
    "read all columns",
    "read 2 columns",
    results,
    "ReadSchema shrinks to customer_id, quantity (gain grows with column count and width)",
)

spark.stop()
