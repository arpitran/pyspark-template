"""05 - Window functions: dedupe, rank, running totals, lag, sessionization.

Run:   uv run python examples/write/05_window_functions.py

Look for:
  - Every input row stays in the output (unlike groupBy).
  - row_number for "latest record per key" dedupes. It is the most common window in real pipelines.
  - Sessionization with lag + running sum, no UDF and no self-join.
  - In the plan: Window needs an Exchange (shuffle) on the partition key plus a Sort.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.transforms.cleaning import split_valid_and_corrupt
from pyspark_template.transforms.enrich import add_order_revenue
from pyspark_template.transforms.sessions import sessionize_clicks, summarize_sessions
from pyspark_template.transforms.windows import (
    days_since_previous_order,
    latest_order_per_customer,
    rank_products_in_category,
    running_spend,
)

spark = build_spark("05-windows")

orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))
# Work on a few customers so the output is readable.
some_orders = orders.filter(F.col("customer_id").between(2, 4))
some_with_revenue = add_order_revenue(some_orders, products)

banner("Latest order per customer (row_number, then keep row 1)")
latest = latest_order_per_customer(orders)
latest.filter(F.col("customer_id") <= 5).orderBy("customer_id").show()
print(f"Distinct customers with orders: {latest.count():,}")

banner("Running revenue per customer")
running_spend(some_with_revenue).select("customer_id", "order_ts", "revenue", "running_revenue").orderBy(
    "customer_id", "order_ts"
).show(10)

banner("Days since previous order (lag)")
days_since_previous_order(some_orders).select("customer_id", "order_ts", "days_since_previous").orderBy(
    "customer_id", "order_ts"
).show(10)

banner("Top 3 products per category by revenue (rank and dense_rank)")
all_with_revenue = add_order_revenue(orders, products)
product_revenue = all_with_revenue.groupBy("category", "product_id").agg(
    F.round(F.sum("revenue"), 2).alias("revenue")
)
ranked = rank_products_in_category(product_revenue)
ranked.filter(F.col("rank") <= 3).orderBy("category", "rank").show(18)

banner("Sessionize clicks (30 minute gap)")
raw_clicks = spark.read.schema(schemas.CLICK_EVENTS).json(str(DATA_DIR / "clicks"))
valid_clicks, _ = split_valid_and_corrupt(raw_clicks)
sessions = summarize_sessions(sessionize_clicks(valid_clicks, gap_minutes=30))
sessions.orderBy("customer_id", "session_start").show(8)
sessions.agg(
    F.count("*").alias("sessions"),
    F.round(F.avg("event_count"), 2).alias("avg_events_per_session"),
    F.round(F.avg("duration_seconds") / 60, 1).alias("avg_minutes"),
).show()

banner("The plan: Window = shuffle by customer_id + sort inside each partition")
latest.explain("formatted")

spark.stop()
