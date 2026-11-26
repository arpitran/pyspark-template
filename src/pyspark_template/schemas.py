"""Explicit schemas for every dataset.

Why explicit: inferSchema reads the data twice and guesses types from a sample.
A wrong guess (an id read as int, then a later file has "A12") fails late or
silently turns values into null. A fixed schema fails fast and is cheaper.
"""

from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

CUSTOMERS = StructType(
    [
        StructField("customer_id", LongType(), nullable=False),
        StructField("name", StringType()),
        StructField("email", StringType()),
        StructField("country", StringType()),
        StructField("signup_date", DateType()),
        StructField("tier", StringType()),
    ]
)

PRODUCTS = StructType(
    [
        StructField("product_id", LongType(), nullable=False),
        StructField("product_name", StringType()),
        StructField("category", StringType()),
        StructField("unit_price", DoubleType()),
    ]
)

ORDERS = StructType(
    [
        StructField("order_id", LongType(), nullable=False),
        StructField("customer_id", LongType()),
        StructField("product_id", LongType()),
        StructField("quantity", IntegerType()),
        StructField("order_ts", TimestampType()),
        StructField("status", StringType()),
        StructField("order_date", DateType()),
    ]
)

# Raw clickstream line. "properties" is left as a JSON string on purpose:
# the transformations example parses it with from_json.
CLICK_EVENTS = StructType(
    [
        StructField("event_id", StringType()),
        StructField("customer_id", LongType()),
        StructField("event_type", StringType()),
        StructField("event_ts", TimestampType()),
        StructField("page", StringType()),
        StructField("properties", StringType()),
        StructField("_corrupt_record", StringType()),
    ]
)

CLICK_PROPERTIES = StructType(
    [
        StructField("device", StringType()),
        StructField("referrer", StringType()),
        StructField("product_ids", StringType()),
    ]
)
