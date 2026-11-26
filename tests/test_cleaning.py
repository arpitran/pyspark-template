from datetime import datetime

from pyspark.sql import Row
from pyspark.testing import assertDataFrameEqual

from pyspark_template.transforms.cleaning import (
    cast_orders_safely,
    parse_click_properties,
    split_valid_and_corrupt,
)


def test_split_valid_and_corrupt_separates_rows(spark):
    raw = spark.createDataFrame(
        [("e1", None), (None, '{"event_id": "e2", "cust')],
        "event_id string, _corrupt_record string",
    )
    valid, corrupt = split_valid_and_corrupt(raw)
    assertDataFrameEqual(valid, [Row(event_id="e1")])
    assertDataFrameEqual(corrupt, [Row(_corrupt_record='{"event_id": "e2", "cust')])


def test_parse_click_properties_explodes_products_and_keeps_empty_events(spark):
    ts = datetime(2026, 3, 1, 10, 0, 0)
    clicks = spark.createDataFrame(
        [
            (
                "e1",
                7,
                "page_view",
                ts,
                "/product",
                '{"device": "ios", "referrer": "google", "product_ids": "5,9"}',
            ),
            ("e2", 7, "search", ts, "/search", '{"device": "web", "referrer": "direct", "product_ids": ""}'),
        ],
        "event_id string, customer_id long, event_type string, event_ts timestamp, page string, properties string",
    )
    result = parse_click_properties(clicks).select("event_id", "device", "product_id")
    expected = [
        Row(event_id="e1", device="ios", product_id=5),
        Row(event_id="e1", device="ios", product_id=9),
        Row(event_id="e2", device="web", product_id=None),
    ]
    assertDataFrameEqual(result, expected)


def test_cast_orders_safely_flags_bad_values_instead_of_hiding_them(spark):
    raw = spark.createDataFrame(
        [
            ("1", "10", "3", "2", "2026-01-05 10:00:00", "placed"),
            ("2", "C-11", "3", "two", "2026-01-05 11:00:00", "placed"),
        ],
        "order_id string, customer_id string, product_id string, quantity string, order_ts string, status string",
    )
    result = cast_orders_safely(raw).select("order_id", "customer_id", "quantity", "has_bad_value")
    expected = [
        Row(order_id=1, customer_id=10, quantity=2, has_bad_value=False),
        Row(order_id=2, customer_id=None, quantity=None, has_bad_value=True),
    ]
    assertDataFrameEqual(result, expected)
