"""Window functions: rank, dedupe, running totals, previous row.

A window function looks at a group of rows (the partition) without
collapsing them the way groupBy does. Every input row stays in the output.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def latest_order_per_customer(orders: DataFrame) -> DataFrame:
    """Keep only each customer's most recent order.

    row_number gives 1, 2, 3 even on ties, so exactly one row survives.
    order_id is the tie-breaker so the result is the same on every run.
    """
    newest_first = Window.partitionBy("customer_id").orderBy(
        F.col("order_ts").desc(), F.col("order_id").desc()
    )
    numbered = orders.withColumn("row_number", F.row_number().over(newest_first))
    return numbered.filter(F.col("row_number") == 1).drop("row_number")


def running_spend(orders_with_revenue: DataFrame) -> DataFrame:
    """Cumulative revenue per customer in order-time order.

    rowsBetween(unboundedPreceding, currentRow) means "every row so far".
    Without it, an ordered window defaults to a RANGE frame, which lumps
    together rows that share the same order_ts.
    """
    so_far = (
        Window.partitionBy("customer_id")
        .orderBy("order_ts", "order_id")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    return orders_with_revenue.withColumn("running_revenue", F.round(F.sum("revenue").over(so_far), 2))


def rank_products_in_category(product_revenue: DataFrame) -> DataFrame:
    """Rank products by revenue inside their category.

    rank:        1, 2, 2, 4  (gap after a tie)
    dense_rank:  1, 2, 2, 3  (no gap)
    """
    by_revenue = Window.partitionBy("category").orderBy(F.col("revenue").desc())
    return product_revenue.withColumn("rank", F.rank().over(by_revenue)).withColumn(
        "dense_rank", F.dense_rank().over(by_revenue)
    )


def days_since_previous_order(orders: DataFrame) -> DataFrame:
    """lag() reads the previous row in the window. The first order has no previous row, so it is null."""
    in_order = Window.partitionBy("customer_id").orderBy("order_ts", "order_id")
    previous_ts = F.lag("order_ts").over(in_order)
    return orders.withColumn("days_since_previous", F.date_diff(F.col("order_ts"), previous_ts))
