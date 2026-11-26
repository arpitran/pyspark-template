"""09 - Structured Streaming on files: watermark, windows, checkpoints, foreachBatch.

Run:   uv run python examples/write/09_streaming_watermark_foreachbatch.py

No Kafka needed: the file source treats each new file in a folder as new data.
(For a Kafka source see the companion repo kafka-streaming-pipeline.)

Look for:
  - trigger(availableNow=True): process everything available, then stop. Batch-like cost, streaming bookkeeping.
  - The checkpoint remembers which files were done. Run again and only the new file is read.
  - The watermark tells Spark how late data may arrive, so it can drop old window state.
  - foreachBatch gives you a normal DataFrame per micro-batch: write it anywhere, MERGE it, run checks.
"""

import shutil

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, OUTPUT_DIR, banner, build_spark

spark = build_spark("09-streaming")
landing = OUTPUT_DIR / "09_landing"
checkpoint = OUTPUT_DIR / "09_checkpoint"
sink = OUTPUT_DIR / "09_counts"
for folder in [landing, checkpoint, sink]:
    shutil.rmtree(folder, ignore_errors=True)
landing.mkdir(parents=True)

click_files = sorted((DATA_DIR / "clicks").glob("*.json"))


def land_files(count, start):
    """Copy click files into the landing folder, as if an upstream system dropped them."""
    copied = 0
    for click_file in click_files[start : start + count]:
        shutil.copy(click_file, landing / click_file.name)
        copied += 1
    print(f"Landed {copied} new file(s)")


batch_log = []


def write_batch(batch_df, batch_id):
    """Called once per micro-batch with a plain (non-streaming) DataFrame."""
    row_count = batch_df.count()
    batch_log.append((batch_id, row_count))
    print(f"  foreachBatch: batch {batch_id} has {row_count} window rows")
    batch_df.write.mode("append").parquet(str(sink))


def run_once():
    stream = (
        spark.readStream.schema(schemas.CLICK_EVENTS)
        .option("maxFilesPerTrigger", 5)
        .json(str(landing))
        .filter(F.col("_corrupt_record").isNull())
    )
    windowed = (
        stream.withWatermark("event_ts", "30 minutes")
        .groupBy(F.window("event_ts", "1 hour"), "event_type")
        .count()
        .select(F.col("window.start").alias("window_start"), "event_type", "count")
    )
    query = (
        windowed.writeStream.outputMode("append")
        .foreachBatch(write_batch)
        .option("checkpointLocation", str(checkpoint))
        .trigger(availableNow=True)
        .start()
    )
    query.awaitTermination()
    for progress in query.recentProgress:
        watermark = progress["eventTime"].get("watermark", "not set")
        late_rows = 0
        for operator in progress["stateOperators"]:
            late_rows += operator.get("numRowsDroppedByWatermark", 0)
        print(
            f"  batch {progress['batchId']}: input rows {progress['numInputRows']}, "
            f"watermark {watermark}, late rows dropped {late_rows}"
        )


banner("Run 1: 10 files waiting, maxFilesPerTrigger=5")
land_files(10, start=0)
run_once()
print("Expect 2 batches with data. Spark may add a no-data batch whose only job is to move the watermark.")
print("Batch 0 emits nothing: the watermark is still unset while it runs, so no window can be closed yet.")

banner("Run 2: same checkpoint, 1 new file -> only that file is read")
land_files(1, start=10)
run_once()
print("The new file's events cover the same 3 days as the old ones. Rows older than the watermark are late")
print("and dropped (see 'late rows dropped'). No window closes because the watermark did not move.")
print("That is the watermark doing its job:")
print("it trades late data for bounded state. Give it a longer delay if late data matters more than memory.")

banner("What the checkpoint stores")
for item in sorted(checkpoint.iterdir()):
    print(" ", item.name)

banner("Output written by foreachBatch")
print("Append mode with a watermark only emits a window after the watermark passes its end,")
print("so the newest windows stay in state until later data arrives.")
results = spark.read.parquet(str(sink))
print(f"Windows written: {results.count()}")
results.orderBy("window_start", "event_type").show(8)

spark.stop()
