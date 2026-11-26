"""06 - Spark SQL: temp views, CTEs, and proof that SQL and DataFrame code compile to the same plan.

Run:   uv run python examples/write/06_spark_sql_temp_views.py

Look for:
  - createOrReplaceTempView makes a DataFrame queryable by name for this session only.
  - The SQL query and the DataFrame query produce the same optimized plan. Pick whichever reads better.
  - Parameterized SQL (args=...) instead of f-strings: no SQL injection, no quoting bugs.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark

spark = build_spark("06-spark-sql")

spark.read.parquet(str(DATA_DIR / "orders")).createOrReplaceTempView("orders")
spark.read.option("header", True).schema(schemas.PRODUCTS).csv(
    str(DATA_DIR / "products.csv")
).createOrReplaceTempView("products")
spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(
    str(DATA_DIR / "customers.csv")
).createOrReplaceTempView("customers")

banner("SQL with a CTE and a window: each country's top category by revenue")
top_category_sql = """
WITH revenue AS (
    SELECT c.country, p.category, SUM(o.quantity * p.unit_price) AS revenue
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status <> 'cancelled'
    GROUP BY c.country, p.category
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY revenue DESC) AS position
    FROM revenue
)
SELECT country, category, ROUND(revenue, 0) AS revenue
FROM ranked
WHERE position = 1
ORDER BY revenue DESC
"""
spark.sql(top_category_sql).show()

banner("Parameterized SQL (named parameters)")
spark.sql(
    "SELECT status, COUNT(*) AS orders FROM orders WHERE order_date BETWEEN :start AND :end GROUP BY status ORDER BY status",
    args={"start": "2026-02-01", "end": "2026-02-07"},
).show()

banner("Same question two ways: SQL vs DataFrame API")
sql_version = spark.sql(
    "SELECT product_id, SUM(quantity) AS units FROM orders WHERE status = 'delivered' GROUP BY product_id"
)
dataframe_version = (
    spark.table("orders")
    .filter(F.col("status") == "delivered")
    .groupBy("product_id")
    .agg(F.sum("quantity").alias("units"))
)
sql_plan = sql_version._jdf.queryExecution().optimizedPlan().toString()
dataframe_plan = dataframe_version._jdf.queryExecution().optimizedPlan().toString()


def strip_expression_ids(plan):
    """Plans name columns like units#123. The numbers differ between queries, so remove them before comparing."""
    import re

    return re.sub(r"#\d+L?", "", plan)


same = strip_expression_ids(sql_plan) == strip_expression_ids(dataframe_plan)
print("Optimized plans identical (ignoring column ids):", same)
print(strip_expression_ids(sql_plan))

banner("Use SQL functions from the DataFrame API with expr / selectExpr")
spark.table("orders").selectExpr(
    "order_id", "date_format(order_ts, 'EEEE') AS weekday", "quantity * 2 AS double_qty"
).show(3)

spark.stop()
