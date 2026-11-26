"""07 - The small-files problem, and compacting it away.

Run:   uv run python examples/optimize/07_small_files_compaction.py

A table made of thousands of tiny files is slow to read: each file costs a file-system
listing, an open, and a Parquet footer read, no matter how few rows it holds. Streaming jobs and
"repartition(200) then partitionBy(date)" writes are the usual cause.

Look for:
  - Same rows, same query, very different read time.
  - Compaction = read the table and rewrite it with sensible file sizes (here: one file per day).
  - On Delta Lake the same job is one command: OPTIMIZE <table> (optionally ZORDER BY a filter column).
"""

import shutil

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark
from pyspark_template.timing import measure, record_benchmark

spark = build_spark("optimize-07-small-files")
orders = spark.read.parquet(str(DATA_DIR / "orders"))
one_month = orders.filter(F.col("order_date").between("2026-01-01", "2026-01-31"))

small_files = OUTPUT_DIR / "optimize07_small_files"
compacted = OUTPUT_DIR / "optimize07_compacted"
for folder in [small_files, compacted]:
    shutil.rmtree(folder, ignore_errors=True)


def count_files(path):
    total = 0
    for _ in path.rglob("*.parquet"):
        total += 1
    return total


banner("Make the problem: 200 tasks each write a file into every day folder")
one_month.repartition(200).write.partitionBy("order_date").parquet(str(small_files))
small_file_count = count_files(small_files)
row_count = one_month.count()
print(f"Files: {small_file_count:,} for {row_count:,} rows")

banner("Compact: one file per day")
spark.read.parquet(str(small_files)).repartition("order_date").write.partitionBy("order_date").parquet(
    str(compacted)
)
compacted_file_count = count_files(compacted)
print(f"Files after compaction: {compacted_file_count:,}")


def daily_report(path):
    table = spark.read.parquet(str(path))
    return table.groupBy("order_date").agg(F.sum("quantity"), F.countDistinct("customer_id")).collect()


results = {}
banner("Read the same month both ways (includes listing the files)")
before_label = f"{small_file_count:,} small files"
after_label = f"{compacted_file_count:,} compacted files"
measure(before_label, lambda: daily_report(small_files), results)
measure(after_label, lambda: daily_report(compacted), results)
record_benchmark(
    f"07 small files (1 month, {row_count:,} rows)",
    before_label,
    after_label,
    results,
    f"{small_file_count // compacted_file_count}x fewer files: fewer opens, footers and tiny tasks",
)

banner("Settings that soften the problem on read")
print(
    "spark.sql.files.maxPartitionBytes (128MB): target bytes per read task; small files get packed together."
)
print("spark.sql.files.openCostInBytes (4MB): the 'cost' Spark assumes per file when packing.")
print("These reduce the task count, but every file is still listed and opened. Compaction fixes the cause.")

spark.stop()
