"""Joins and aggregations on orders."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_order_revenue(orders: DataFrame, products: DataFrame) -> DataFrame:
    """Attach product details and compute revenue per order line.

    Inner join on purpose: an order for an unknown product is a data problem
    we want to find (see the left_anti join in examples/write/03), not a row
    with null revenue.
    """
    joined = orders.join(
        products.select("product_id", "category", "unit_price"), on="product_id", how="inner"
    )
    return joined.withColumn("revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))


def customer_spend_summary(orders_with_revenue: DataFrame) -> DataFrame:
    """One row per customer: order count, total and average spend, first and last order."""
    return orders_with_revenue.groupBy("customer_id").agg(
        F.countDistinct("order_id").alias("order_count"),
        F.round(F.sum("revenue"), 2).alias("total_revenue"),
        F.round(F.avg("revenue"), 2).alias("avg_revenue"),
        F.min("order_ts").alias("first_order_ts"),
        F.max("order_ts").alias("last_order_ts"),
    )


def revenue_by_category_and_status(orders_with_revenue: DataFrame, statuses) -> DataFrame:
    """Pivot: one row per category, one column per order status.

    Passing the list of statuses up front saves Spark a pass over the data
    to find the distinct values, and fixes the column order.
    """
    pivoted = (
        orders_with_revenue.groupBy("category")
        .pivot("status", statuses)
        .agg(F.round(F.sum("revenue"), 2))
        .na.fill(0.0)
    )
    return pivoted.orderBy("category")
