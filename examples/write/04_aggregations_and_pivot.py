"""04 - groupBy aggregations, pivot, rollup and cube.

Run:   uv run python examples/write/04_aggregations_and_pivot.py

Look for:
  - Several aggregates in one agg() call: one pass over the data, not one per metric.
  - pivot with an explicit value list (saves a pass to find the distinct values).
  - rollup adds subtotal rows (category total, grand total). The null in a key column marks the subtotal.
  - approx_count_distinct: much cheaper than countDistinct on big data, within ~5% by default.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.transforms.enrich import (
    add_order_revenue,
    customer_spend_summary,
    revenue_by_category_and_status,
)

spark = build_spark("04-aggregations")

orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))
orders_with_revenue = add_order_revenue(orders, products)

banner("Many metrics, one agg() call")
daily = (
    orders_with_revenue.groupBy("order_date")
    .agg(
        F.count("*").alias("orders"),
        F.round(F.sum("revenue"), 2).alias("revenue"),
        F.round(F.avg("revenue"), 2).alias("avg_order_value"),
        F.approx_count_distinct("customer_id").alias("approx_customers"),
        F.sum(F.when(F.col("status") == "cancelled", 1).otherwise(0)).alias("cancelled"),
    )
    .orderBy("order_date")
)
daily.show(5)

banner("Top customers (transforms/enrich.py: customer_spend_summary)")
customer_spend_summary(orders_with_revenue).orderBy(F.col("total_revenue").desc()).show(5)
print("Customer 1 is the 'hot key' built into the data: ~30% of all orders. optimize/02 deals with it.")

banner("Pivot: revenue by category x status")
statuses = ["placed", "shipped", "delivered", "cancelled"]
revenue_by_category_and_status(orders_with_revenue, statuses).show()

banner("rollup: category + status, with subtotals and a grand total")
rolled = (
    orders_with_revenue.rollup("category", "status")
    .agg(F.round(F.sum("revenue"), 0).alias("revenue"))
    .orderBy(F.col("category").asc_nulls_last(), F.col("status").asc_nulls_last())
)
rolled.show(12)
print("status = null -> subtotal for that category.  category = null -> grand total.")

banner("countDistinct vs approx_count_distinct")
exact_and_approx = orders.agg(
    F.countDistinct("customer_id").alias("exact"),
    F.approx_count_distinct("customer_id").alias("approx_default_5pct"),
    F.approx_count_distinct("customer_id", 0.01).alias("approx_1pct"),
)
exact_and_approx.show()

spark.stop()
