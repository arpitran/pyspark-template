from datetime import datetime

from pyspark.sql import Row
from pyspark.testing import assertDataFrameEqual

from pyspark_template.transforms.windows import (
    days_since_previous_order,
    latest_order_per_customer,
    rank_products_in_category,
    running_spend,
)


def make_orders(spark):
    return spark.createDataFrame(
        [
            (1, 100, datetime(2026, 1, 1), 10.0),
            (2, 100, datetime(2026, 1, 4), 5.0),
            (3, 100, datetime(2026, 1, 4), 1.0),  # same timestamp as order 2: order_id breaks the tie
            (4, 200, datetime(2026, 1, 2), 7.0),
        ],
        "order_id long, customer_id long, order_ts timestamp, revenue double",
    )


def test_latest_order_per_customer_is_deterministic_on_ties(spark):
    result = latest_order_per_customer(make_orders(spark)).select("customer_id", "order_id")
    assertDataFrameEqual(result, [Row(customer_id=100, order_id=3), Row(customer_id=200, order_id=4)])


def test_running_spend_counts_tied_rows_one_at_a_time(spark):
    result = running_spend(make_orders(spark)).select("order_id", "running_revenue")
    expected = [
        Row(order_id=1, running_revenue=10.0),
        Row(order_id=2, running_revenue=15.0),
        Row(order_id=3, running_revenue=16.0),
        Row(order_id=4, running_revenue=7.0),
    ]
    assertDataFrameEqual(result, expected)


def test_rank_vs_dense_rank(spark):
    product_revenue = spark.createDataFrame(
        [("books", 1, 50.0), ("books", 2, 40.0), ("books", 3, 40.0), ("books", 4, 10.0)],
        "category string, product_id long, revenue double",
    )
    result = rank_products_in_category(product_revenue).select("product_id", "rank", "dense_rank")
    expected = [
        Row(product_id=1, rank=1, dense_rank=1),
        Row(product_id=2, rank=2, dense_rank=2),
        Row(product_id=3, rank=2, dense_rank=2),
        Row(product_id=4, rank=4, dense_rank=3),
    ]
    assertDataFrameEqual(result, expected)


def test_days_since_previous_order(spark):
    result = days_since_previous_order(make_orders(spark)).select("order_id", "days_since_previous")
    expected = [
        Row(order_id=1, days_since_previous=None),
        Row(order_id=2, days_since_previous=3),
        Row(order_id=3, days_since_previous=0),
        Row(order_id=4, days_since_previous=None),
    ]
    assertDataFrameEqual(result, expected)
