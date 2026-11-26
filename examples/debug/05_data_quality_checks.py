"""05 - Data-quality checks as a pipeline gate (src/pyspark_template/quality.py).

Run:   uv run python examples/debug/05_data_quality_checks.py

Look for:
  - Uniqueness, null and accepted-value checks share ONE aggregation pass over the data.
  - A failing check raises DataQualityError, so bad data never reaches the next step.
  - Many "wrong numbers" bugs are really duplicate keys after a join. Check the grain before you sum.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.quality import (
    DataQualityError,
    check_columns,
    check_foreign_key,
    check_not_empty,
    run_checks,
)
from pyspark_template.session import DATA_DIR, banner, build_spark

spark = build_spark("debug-05-quality")

orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))
customers = spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(str(DATA_DIR / "customers.csv"))

banner("Checks on the clean orders table")
results = [check_not_empty(orders)]
column_results = check_columns(
    orders,
    key_column="order_id",
    not_null_columns=["customer_id", "product_id", "order_ts"],
    accepted_values={"status": ["placed", "shipped", "delivered", "cancelled"]},
)
for result in column_results:
    results.append(result)
results.append(check_foreign_key(orders, products, "product_id"))
results.append(check_foreign_key(orders, customers, "customer_id"))
run_checks(results)

banner("A classic bug: a join that multiplies rows")
# Someone adds a "product price history" table with two prices for some products,
# then joins it to orders. Revenue silently goes up.
price_history = products.select("product_id", "unit_price").union(
    products.filter("product_id <= 50").select("product_id", (F.col("unit_price") * 1.1).alias("unit_price"))
)
orders_with_price = orders.join(price_history, "product_id")
print(f"orders: {orders.count():,}   after join: {orders_with_price.count():,}")
try:
    run_checks(check_columns(orders_with_price, key_column="order_id", not_null_columns=["unit_price"]))
except DataQualityError as error:
    print("Pipeline stopped:", error)
print("Fix: make the lookup one row per key first (latest price via row_number), then join.")

banner("Unexpected values sneak in")
with_typo = orders.withColumn(
    "status", F.when(F.col("order_id") % 100_000 == 0, F.lit("shiped")).otherwise(F.col("status"))
)
try:
    run_checks(
        check_columns(
            with_typo,
            "order_id",
            [],
            accepted_values={"status": ["placed", "shipped", "delivered", "cancelled"]},
        )
    )
except DataQualityError as error:
    print("Pipeline stopped:", error)

spark.stop()
