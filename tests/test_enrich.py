from datetime import datetime

from pyspark.sql import Row
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual, assertSchemaEqual

from pyspark_template.transforms.enrich import (
    add_order_revenue,
    customer_spend_summary,
    revenue_by_category_and_status,
)


def make_orders(spark):
    return spark.createDataFrame(
        [
            (1, 100, 1, 2, datetime(2026, 1, 1, 9), "placed"),
            (2, 100, 2, 1, datetime(2026, 1, 2, 9), "shipped"),
            (3, 200, 1, 1, datetime(2026, 1, 3, 9), "shipped"),
            (4, 200, 999, 1, datetime(2026, 1, 4, 9), "placed"),  # unknown product
        ],
        "order_id long, customer_id long, product_id long, quantity int, order_ts timestamp, status string",
    )


def make_products(spark):
    return spark.createDataFrame(
        [(1, "books", 10.0), (2, "toys", 5.5)],
        "product_id long, category string, unit_price double",
    )


def test_add_order_revenue_multiplies_and_drops_unknown_products(spark):
    result = add_order_revenue(make_orders(spark), make_products(spark)).select("order_id", "revenue")
    assertDataFrameEqual(
        result, [Row(order_id=1, revenue=20.0), Row(order_id=2, revenue=5.5), Row(order_id=3, revenue=10.0)]
    )


def test_customer_spend_summary(spark):
    with_revenue = add_order_revenue(make_orders(spark), make_products(spark))
    result = customer_spend_summary(with_revenue).select("customer_id", "order_count", "total_revenue")
    assertDataFrameEqual(
        result,
        [
            Row(customer_id=100, order_count=2, total_revenue=25.5),
            Row(customer_id=200, order_count=1, total_revenue=10.0),
        ],
    )


def test_revenue_pivot_has_one_column_per_status_and_zero_fills(spark):
    with_revenue = add_order_revenue(make_orders(spark), make_products(spark))
    result = revenue_by_category_and_status(with_revenue, ["placed", "shipped"])

    expected_schema = StructType(
        [
            StructField("category", StringType()),
            StructField("placed", DoubleType()),
            StructField("shipped", DoubleType()),
        ]
    )
    assertSchemaEqual(result.schema, expected_schema)
    assertDataFrameEqual(
        result,
        [Row(category="books", placed=20.0, shipped=10.0), Row(category="toys", placed=0.0, shipped=5.5)],
    )
