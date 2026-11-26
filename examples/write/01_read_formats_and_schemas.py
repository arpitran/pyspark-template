"""01 - Reading CSV, JSON and Parquet, and why explicit schemas win.

Run:   uv run python examples/write/01_read_formats_and_schemas.py

Look for:
  - inferSchema guesses customer_id as a string because a few rows hold "C-123".
  - The explicit schema reads the same file in one pass. Bad values become null (PERMISSIVE)
    or stop the job (FAILFAST): you choose.
  - Parquet stores its own schema, so it needs no options at all.
"""

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark, short_error

spark = build_spark("01-read-formats")

banner("CSV with inferSchema: Spark reads the file twice and guesses types")
inferred = (
    spark.read.option("header", True).option("inferSchema", True).csv(str(DATA_DIR / "orders_sample.csv"))
)
inferred.printSchema()
print("customer_id and quantity became strings: one bad value spoils the whole column.")

banner("CSV with an explicit schema (PERMISSIVE, the default mode)")
orders_sample_schema = schemas.ORDERS
explicit = (
    spark.read.option("header", True)
    .option("mode", "PERMISSIVE")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
    .schema(orders_sample_schema)
    .csv(str(DATA_DIR / "orders_sample.csv"))
)
explicit.printSchema()
rows_with_null_quantity = explicit.filter("quantity IS NULL").count()
print(f"Rows where quantity could not be parsed and became null: {rows_with_null_quantity}")

banner("Same read in FAILFAST mode: the first bad value stops the job")
failfast = (
    spark.read.option("header", True)
    .option("mode", "FAILFAST")
    .schema(orders_sample_schema)
    .csv(str(DATA_DIR / "orders_sample.csv"))
)
print(f"count() in FAILFAST mode returned {failfast.count()} with no error. Why?")
print("CSV column pruning: count() needs no columns, so Spark never parses quantity and never sees 'two'.")
try:
    failfast.collect()
except Exception as error:
    print("collect() parses every column and FAILFAST raised:", short_error(error))
print("Use FAILFAST when bad input must never be loaded; PERMISSIVE plus a check when it can be quarantined.")

banner("JSON lines with a schema and a corrupt-record column")
clicks = (
    spark.read.schema(schemas.CLICK_EVENTS)
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .json(str(DATA_DIR / "clicks"))
)
clicks.show(3, truncate=60)
print(f"Click files read: {len(clicks.inputFiles())}")

banner("Parquet: schema comes from the files, partition column comes from the folder names")
orders = spark.read.parquet(str(DATA_DIR / "orders"))
orders.printSchema()
print(f"Orders: {orders.count():,} rows")
print("order_date is not stored inside the files. Spark rebuilt it from folders like order_date=2026-01-01/.")

banner("Customers and products CSV with explicit schemas")
customers = spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(str(DATA_DIR / "customers.csv"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))
customers.show(3)
products.show(3)

spark.stop()
