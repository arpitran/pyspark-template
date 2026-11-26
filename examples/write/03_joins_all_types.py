"""03 - Every join type on small, readable tables, then on real data.

Run:   uv run python examples/write/03_joins_all_types.py

Look for:
  - inner / left / right / full outer: which unmatched rows survive.
  - left_semi: "rows in A that have a match in B" without adding B's columns (like SQL EXISTS).
  - left_anti: "rows in A with no match in B" (like NOT EXISTS). The best way to find orphans.
  - Joining on a column name (on="id") keeps one id column. Joining on an expression keeps both.
"""

from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark

spark = build_spark("03-joins")

customers = spark.createDataFrame([(1, "Asha"), (2, "Ben"), (3, "Chen")], "customer_id long, name string")
orders = spark.createDataFrame(
    [(10, 1, 25.0), (11, 1, 5.0), (12, 2, 40.0), (13, 9, 99.0)],
    "order_id long, customer_id long, amount double",
)

print("customers: 1 Asha, 2 Ben, 3 Chen (no orders)")
print("orders:    10 and 11 for Asha, 12 for Ben, 13 for customer 9 (does not exist)")

join_types = ["inner", "left", "right", "full", "left_semi", "left_anti"]
for join_type in join_types:
    banner(f"customers {join_type} join orders")
    result = customers.join(orders, on="customer_id", how=join_type)
    result.orderBy("customer_id").show()

banner("cross join: every pair. Rarely wanted; Spark makes you ask for it by name")
customers.crossJoin(spark.createDataFrame([("web",), ("ios",)], "channel string")).show()

banner("Join on an expression keeps both key columns - select or drop one to avoid ambiguity")
by_expression = customers.alias("c").join(
    orders.alias("o"), F.col("c.customer_id") == F.col("o.customer_id"), "inner"
)
by_expression.printSchema()
by_expression.select("c.customer_id", "c.name", "o.amount").show()

banner("Real data: find orders whose product does not exist (left_anti)")
real_orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))
orphans = real_orders.join(products, on="product_id", how="left_anti")
print(f"Orders with an unknown product: {orphans.count()}  (0 means referential integrity holds)")

banner("Real data: customers who never ordered (left_anti the other way)")
all_customers = (
    spark.read.option("header", True).schema(schemas.CUSTOMERS).csv(str(DATA_DIR / "customers.csv"))
)
never_ordered = all_customers.join(
    real_orders.select("customer_id").distinct(), on="customer_id", how="left_anti"
)
print(f"Customers with no orders: {never_ordered.count():,}")

spark.stop()
