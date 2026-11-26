"""03 - Errors you will hit, reproduced on purpose, each with its fix.

Run:   uv run python examples/debug/03_common_errors_and_fixes.py

Each section: reproduce -> catch -> print the error -> show the fix.
  1. AMBIGUOUS_REFERENCE after a join
  2. Casting bad strings: ANSI mode error vs silent null vs try_cast
  3. Lazy evaluation: the error appears at the action, not on the line with the bug
  4. Python UDF with the wrong return type: silent nulls
  5. collect() on a big DataFrame: spark.driver.maxResultSize
  6. Column-not-found: caught early, at analysis time
"""

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from pyspark_template.session import DATA_DIR, banner, build_spark, short_error

# A small maxResultSize so section 5 fails fast instead of filling memory.
spark = build_spark("debug-03-errors", extra_conf={"spark.driver.maxResultSize": "8m"})

orders = spark.read.parquet(str(DATA_DIR / "orders"))

banner("1. AMBIGUOUS_REFERENCE: both sides of the join have customer_id")
order_counts = orders.groupBy("customer_id").agg(F.count("*").alias("order_count"))
big_spenders = orders.filter("quantity = 5").select("customer_id", "order_id")
joined = big_spenders.join(order_counts, big_spenders.customer_id == order_counts.customer_id)
try:
    joined.select("customer_id", "order_count").show(2)
except Exception as error:
    print("Raised:", short_error(error))
print("Fix A: join on the column name, which keeps one customer_id:")
big_spenders.join(order_counts, on="customer_id").select("customer_id", "order_count").show(2)
print("Fix B: alias both sides and qualify the column:")
aliased = big_spenders.alias("b").join(
    order_counts.alias("c"), F.col("b.customer_id") == F.col("c.customer_id")
)
aliased.select("b.customer_id", "c.order_count").show(2)

banner("2. Casting bad strings")
raw = spark.createDataFrame([("1",), ("2",), ("two",)], "quantity string")
print("Spark 4 turns ANSI mode on by default: a bad cast is an error, not a null.")
try:
    raw.select(F.col("quantity").cast("int")).collect()
except Exception as error:
    print("ANSI on, cast raised:", short_error(error))
spark.conf.set("spark.sql.ansi.enabled", "false")
print("ANSI off (Spark 3 default): the bad value silently becomes null. Easy to miss:")
raw.select(F.col("quantity").cast("int").alias("quantity")).show()
spark.conf.set("spark.sql.ansi.enabled", "true")
print("Fix: try_cast makes the intent explicit. Keep the raw value so bad rows can be counted:")
raw.select("quantity", F.expr("try_cast(quantity AS INT)").alias("quantity_int")).show()

banner("3. Lazy evaluation: the bug is on one line, the error shows up later")
print("Building the DataFrame does no work, so this line succeeds even though 1/0 is coming:")
risky = orders.withColumn("per_unit", F.lit(10) / (F.col("quantity") - F.col("quantity")))
print("  -> DataFrame defined, no error yet")
try:
    risky.select("per_unit").limit(5).collect()
except Exception as error:
    print("The action raised:", short_error(error))
print("Fix: when an action fails, check every transformation since the last action, not just the last line.")
print("Tip: while debugging, add a temporary .limit(5).collect() after each step to find where it breaks.")

banner("4. Python UDF declared IntegerType but returns a float: nulls, no error")


def half_quantity(quantity):
    return quantity / 2  # a float, e.g. 1.5


half_wrong = F.udf(half_quantity, IntegerType())
orders.select("quantity", half_wrong("quantity").alias("half")).limit(4).show()
print("Every value is null: Spark could not store a Python float in an int column.")
print("Fix: return the declared type, or better, use a built-in (no UDF at all):")
orders.select("quantity", (F.col("quantity") / 2).alias("half")).limit(4).show()

banner("5. collect() pulls every row into the driver")
try:
    rows = orders.collect()
    print(f"collected {len(rows)} rows")
except Exception as error:
    print("Raised:", short_error(error))
print("Fixes: aggregate in Spark first; take(n)/limit(n) to look at rows; write results to files;")
print("toLocalIterator() to stream partitions one at a time when the driver really needs every row.")
first_rows = orders.take(3)
print(f"take(3) is safe: {len(first_rows)} rows")

banner("6. A misspelled column fails immediately, at analysis time (no job runs)")
try:
    orders.select("custmer_id")
except Exception as error:
    print("Raised:", short_error(error))
print("Analysis errors are eager and cheap. Runtime errors (sections 2-3) wait for an action.")

spark.stop()
