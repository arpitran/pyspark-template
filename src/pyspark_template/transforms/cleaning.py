"""Cleaning raw input: bad records, JSON strings, unsafe casts."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark_template.schemas import CLICK_PROPERTIES


def split_valid_and_corrupt(raw_df: DataFrame, corrupt_column="_corrupt_record"):
    """Split a PERMISSIVE read into (valid rows, corrupt rows).

    In PERMISSIVE mode Spark keeps unparseable lines and puts the raw text in
    the corrupt-record column. Every other column is null for those rows.
    """
    # Spark refuses queries that reference only the corrupt column of an un-cached raw
    # JSON/CSV read, so cache first. See examples/debug/02_bad_records_quarantine.py.
    cached = raw_df.cache()
    valid_rows = cached.filter(F.col(corrupt_column).isNull()).drop(corrupt_column)
    corrupt_rows = cached.filter(F.col(corrupt_column).isNotNull()).select(corrupt_column)
    return valid_rows, corrupt_rows


def parse_click_properties(clicks_df: DataFrame) -> DataFrame:
    """Turn the JSON string column "properties" into real columns.

    product_ids arrives as "12,40,7". Split it into an array and explode it so
    each viewed product gets its own row. explode_outer keeps events with no
    products (plain explode would drop them).
    """
    parsed = clicks_df.withColumn("props", F.from_json("properties", CLICK_PROPERTIES))
    with_columns = parsed.select(
        "event_id",
        "customer_id",
        "event_type",
        "event_ts",
        "page",
        F.col("props.device").alias("device"),
        F.col("props.referrer").alias("referrer"),
        F.when(F.col("props.product_ids") == "", F.lit(None))
        .otherwise(F.split("props.product_ids", ",").cast("array<bigint>"))
        .alias("product_id_list"),
    )
    # Cast the array before exploding. explode(...).cast(...) fails with
    # UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS: a generator must be the outermost expression.
    exploded = with_columns.withColumn("product_id", F.explode_outer("product_id_list"))
    return exploded.drop("product_id_list")


def cast_orders_safely(raw_orders: DataFrame) -> DataFrame:
    """Cast string columns to real types without losing track of bad values.

    A plain .cast("int") turns "two" into null with no error, so bad data
    disappears silently. try_cast does the same but we keep the raw value and
    add a flag column, so bad rows can be counted and quarantined.
    """
    casted = raw_orders.select(
        F.expr("try_cast(order_id AS BIGINT)").alias("order_id"),
        F.expr("try_cast(customer_id AS BIGINT)").alias("customer_id"),
        F.expr("try_cast(product_id AS BIGINT)").alias("product_id"),
        F.expr("try_cast(quantity AS INT)").alias("quantity"),
        F.expr("try_cast(order_ts AS TIMESTAMP)").alias("order_ts"),
        F.col("status"),
        F.col("customer_id").alias("raw_customer_id"),
        F.col("quantity").alias("raw_quantity"),
    )
    has_bad_value = (F.col("customer_id").isNull() & F.col("raw_customer_id").isNotNull()) | (
        F.col("quantity").isNull() & F.col("raw_quantity").isNotNull()
    )
    return casted.withColumn("has_bad_value", has_bad_value)
