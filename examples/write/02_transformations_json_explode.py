"""02 - Column transformations: select, withColumn, when, from_json, explode, regex.

Run:   uv run python examples/write/02_transformations_json_explode.py

Look for:
  - from_json turns a JSON string column into a struct you can query with dot notation.
  - explode_outer gives one row per array element and keeps rows with an empty array.
  - Built-in functions (F.when, F.regexp_extract) instead of Python UDFs. See optimize/05 for why.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.transforms.cleaning import parse_click_properties, split_valid_and_corrupt

spark = build_spark("02-transformations")

raw_clicks = (
    spark.read.schema(schemas.CLICK_EVENTS)
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .json(str(DATA_DIR / "clicks"))
)
valid_clicks, corrupt_clicks = split_valid_and_corrupt(raw_clicks)
print(f"Valid click lines: {valid_clicks.count():,}   corrupt lines: {corrupt_clicks.count():,}")

banner("Step 1: the raw properties column is just a string")
valid_clicks.select("event_id", "properties").show(3, truncate=False)

banner("Step 2: from_json + explode_outer (transforms/cleaning.py: parse_click_properties)")
parsed = parse_click_properties(valid_clicks)
parsed.select("event_id", "device", "referrer", "product_id").show(8)
print(f"Rows before explode: {valid_clicks.count():,}   after: {parsed.count():,}")

banner("Step 3: derive columns with when/otherwise, regexp_extract, date functions")
derived = (
    parsed.withColumn(
        "funnel_stage",
        F.when(F.col("event_type") == "checkout", "3-checkout")
        .when(F.col("event_type") == "add_to_cart", "2-cart")
        .otherwise("1-browse"),
    )
    .withColumn("page_section", F.regexp_extract("page", r"^/([a-z]*)", 1))
    .withColumn("page_section", F.when(F.col("page_section") == "", "home").otherwise(F.col("page_section")))
    .withColumn("event_hour", F.hour("event_ts"))
    .withColumn("event_day", F.to_date("event_ts"))
)
derived.select("event_id", "event_type", "funnel_stage", "page", "page_section", "event_hour").show(8)

banner("Step 4: struct and array helpers - collect products per event back into a list")
regrouped = parsed.groupBy("event_id").agg(F.collect_list("product_id").alias("product_ids"))
regrouped.withColumn("product_count", F.size("product_ids")).show(5, truncate=False)

banner("Step 5: inline + map_entries - flatten a map column into key/value rows")
# Metrics often arrive as a JSON map, e.g. {"page_view": 12, "checkout": 1}. Build one to work with.
# Count first: map keys must be unique, so building a map straight from raw events fails with
# DUPLICATED_MAP_KEY as soon as a customer has two events of the same type.
per_type = valid_clicks.groupBy("customer_id", "event_type").count()
as_json_map = per_type.groupBy("customer_id").agg(
    F.to_json(F.map_from_entries(F.collect_list(F.struct("event_type", "count")))).alias("metrics_json")
)
as_json_map.show(3, truncate=False)

parsed_map = as_json_map.withColumn("metrics", F.from_json("metrics_json", "map<string, bigint>"))
flattened = parsed_map.selectExpr("customer_id", "inline(map_entries(metrics))").withColumnRenamed(
    "key", "event_type"
)
flattened.withColumnRenamed("value", "events").orderBy("customer_id", "event_type").show(6)
print("Pattern for metrics stored as a JSON map: from_json to a map, then inline(map_entries(...)) to rows.")

spark.stop()
