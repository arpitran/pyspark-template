"""02 - Bad input records: find them, quarantine them, decide whether to fail the run.

Run:   uv run python examples/debug/02_bad_records_quarantine.py

Look for:
  - The corrupt-record column holds the raw text of every line Spark could not parse.
  - Spark refuses a query that reads ONLY the corrupt column of an uncached raw file. cache() first.
  - input_file_name() tells you which file each bad line came from.
  - A threshold check: a little bad data gets quarantined; a lot of bad data stops the pipeline.
  - Per-file isolation: one broken file should not hide the result of the other 19.
"""

import shutil

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.quality import DataQualityError, check_ratio_at_most, run_checks
from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark, short_error

spark = build_spark("debug-02-bad-records")
quarantine_path = OUTPUT_DIR / "debug02_quarantine"
shutil.rmtree(quarantine_path, ignore_errors=True)

raw = (
    spark.read.schema(schemas.CLICK_EVENTS)
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .json(str(DATA_DIR / "clicks"))
    .withColumn("source_file", F.input_file_name())
)

banner("Gotcha: selecting only the corrupt column of a raw JSON read is not allowed")
try:
    raw.filter(F.col("_corrupt_record").isNotNull()).select("_corrupt_record").count()
    print("No error on this Spark version.")
except Exception as error:
    print("Raised:", short_error(error))
print("Fix: cache (or persist) the parsed DataFrame, then filter it.")

parsed = raw.cache()
bad_lines = parsed.filter(F.col("_corrupt_record").isNotNull())
good_lines = parsed.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

banner("Which files have bad lines, and how many?")
bad_by_file = bad_lines.groupBy(F.regexp_extract("source_file", r"([^/]+)$", 1).alias("file")).count()
bad_by_file.orderBy(F.col("count").desc()).show(5)

banner("What does a bad line look like?")
bad_lines.select("_corrupt_record").show(3, truncate=100)

banner("Quarantine: keep bad lines with their source file for someone to inspect or replay")
bad_lines.select("_corrupt_record", "source_file", F.current_timestamp().alias("quarantined_at")).write.mode(
    "overwrite"
).json(str(quarantine_path))
print(f"Quarantined lines written to {quarantine_path.relative_to(OUTPUT_DIR.parent)}")

banner("Threshold check: fail the run if more than 5% of lines are bad")
total = parsed.count()
bad = bad_lines.count()
run_checks([check_ratio_at_most(bad, total, 0.05, "corrupt_line_ratio")])

banner("The same check with a 1% limit fails, and stops the pipeline")
try:
    run_checks([check_ratio_at_most(bad, total, 0.01, "corrupt_line_ratio")])
except DataQualityError as error:
    print("Pipeline stopped:", error)

banner("Per-file isolation: process files one by one, record failures, continue")
# Four clean files (good lines only) plus one broken file, in one landing folder.
landing = OUTPUT_DIR / "debug02_landing"
shutil.rmtree(landing, ignore_errors=True)
good_lines.drop("source_file").limit(4000).repartition(4).write.json(str(landing))
broken_file = landing / "part-99999-broken.json"
broken_file.write_text('{"event_id": "x1", "customer_id": 5, "event_ty\n')
files_to_load = []
for data_file in sorted(landing.glob("part-*.json")):
    files_to_load.append(str(data_file))

succeeded = []
failed = []
for path in files_to_load:
    try:
        rows = spark.read.schema(schemas.CLICK_EVENTS).option("mode", "FAILFAST").json(path).collect()
        succeeded.append((path.rsplit("/", 1)[-1], len(rows)))
    except Exception as error:
        failed.append((path.rsplit("/", 1)[-1], short_error(error, 120)))

print(f"Succeeded: {len(succeeded)}   failed: {len(failed)}")
for name, message in failed:
    print(f"  FAILED {name}: {message}")
print("One bad file is named and skipped; the good files still load. Reading the whole folder at once")
print("in FAILFAST mode would have failed everything with no clue which file was at fault.")
print("Choose per source: PERMISSIVE + quarantine for logs, FAILFAST for contracts like finance feeds.")

spark.stop()
