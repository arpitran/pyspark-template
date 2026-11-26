"""07 - Writing data: save modes, partitionBy, dynamic partition overwrite, file counts.

Run:   uv run python examples/write/07_writes_partitioning_modes.py

Look for:
  - mode("overwrite") with static partition overwrite wipes the WHOLE table, even partitions you did not write.
  - partitionOverwriteMode=dynamic replaces only the partitions present in the new data.
    This is how you rerun one day of a daily job safely.
  - The number of files written = the number of tasks writing to each partition.
"""

import shutil

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark, short_error

spark = build_spark("07-writes")
target = OUTPUT_DIR / "07_daily_orders"
if target.exists():
    shutil.rmtree(target)


def count_partitions_and_files(path):
    partition_folders = []
    file_count = 0
    for folder in sorted(path.glob("order_date=*")):
        partition_folders.append(folder.name)
        for data_file in folder.glob("*.parquet"):
            file_count += 1
    return partition_folders, file_count


orders = spark.read.parquet(str(DATA_DIR / "orders"))
first_week = orders.filter(F.col("order_date").between("2026-01-01", "2026-01-07"))

banner("Write one week, partitioned by order_date")
first_week.write.mode("overwrite").partitionBy("order_date").parquet(str(target))
folders, files = count_partitions_and_files(target)
print(f"Partitions: {len(folders)}  files: {files}")

banner("Mode 'append' adds files; mode 'errorifexists' (the default) refuses to touch existing data")
try:
    first_week.limit(10).write.partitionBy("order_date").parquet(str(target))
except Exception as error:
    print("errorifexists raised:", short_error(error))

banner("Static overwrite of ONE day wipes the other six days")
corrected_jan_3 = orders.filter(F.col("order_date") == "2026-01-03").withColumn(
    "status", F.lit("reprocessed")
)
corrected_jan_3.write.mode("overwrite").partitionBy("order_date").parquet(str(target))
folders, files = count_partitions_and_files(target)
print(f"Partitions left after static overwrite: {folders}")

banner("Rewrite the week, then dynamic overwrite of one day")
first_week.write.mode("overwrite").partitionBy("order_date").parquet(str(target))
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
corrected_jan_3.write.mode("overwrite").partitionBy("order_date").parquet(str(target))
folders, files = count_partitions_and_files(target)
print(f"Partitions after dynamic overwrite: {len(folders)} (all seven kept)")
spark.read.parquet(str(target)).groupBy("order_date", "status").count().filter(
    "order_date = '2026-01-03'"
).show()
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "static")

banner("File count control: repartition by the partition column before writing")
small_files_target = OUTPUT_DIR / "07_files_check"
first_week.repartition(16).write.mode("overwrite").partitionBy("order_date").parquet(str(small_files_target))
_, files_without = count_partitions_and_files(small_files_target)
first_week.repartition("order_date").write.mode("overwrite").partitionBy("order_date").parquet(
    str(small_files_target)
)
_, files_with = count_partitions_and_files(small_files_target)
print(f"repartition(16) then partitionBy: {files_without} files (up to 16 per day)")
print(f"repartition('order_date') then partitionBy: {files_with} files (one per day)")

banner("maxRecordsPerFile caps file size when one partition is huge")
first_week.repartition("order_date").write.mode("overwrite").option("maxRecordsPerFile", 20_000).partitionBy(
    "order_date"
).parquet(str(small_files_target))
_, capped_files = count_partitions_and_files(small_files_target)
print(f"With maxRecordsPerFile=20000: {capped_files} files")

spark.stop()
