"""Generate the synthetic datasets every example reads.

Run:  make data   (or: uv run python data/generate.py)

What it writes to data/generated/:
  customers.csv          100,000 rows, plain CSV with a header
  products.csv           1,000 rows (small table, the broadcast-join side)
  orders/                5,000,000 rows of Parquet, partitioned by order_date.
                         Customer 1 owns ~30% of the orders, so joins on customer_id are skewed.
  orders_sample.csv      1,000 orders as CSV. Has a few rows with bad values, for the schema demos.
  clicks/                ~50,000 JSON lines in 20 files. ~2% of the lines are malformed.

Everything is seeded, so every run makes the same data.
"""

import csv
import json
import random
import shutil
from datetime import date, datetime, timedelta

from pyspark.sql import functions as F

from pyspark_template.session import DATA_DIR, banner, build_spark

SEED = 42
NUMBER_OF_CUSTOMERS = 100_000
NUMBER_OF_PRODUCTS = 1_000
NUMBER_OF_ORDERS = 5_000_000
HOT_CUSTOMER_SHARE = 0.30
NUMBER_OF_CLICK_FILES = 20
CLICKS_PER_FILE = 2_500
START_DATE = date(2026, 1, 1)
NUMBER_OF_DAYS = 90

COUNTRIES = ["US", "CA", "GB", "DE", "IN", "AU", "FR", "BR"]
TIERS = ["free", "silver", "gold"]
CATEGORIES = ["books", "electronics", "home", "toys", "garden", "sports"]
EVENT_TYPES = ["page_view", "page_view", "page_view", "add_to_cart", "checkout", "search"]
DEVICES = ["ios", "android", "web"]


def write_customers(rng, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_id", "name", "email", "country", "signup_date", "tier"])
        for customer_id in range(1, NUMBER_OF_CUSTOMERS + 1):
            signup = START_DATE - timedelta(days=rng.randint(0, 900))
            writer.writerow(
                [
                    customer_id,
                    f"Customer {customer_id}",
                    f"customer{customer_id}@example.com",
                    rng.choice(COUNTRIES),
                    signup.isoformat(),
                    rng.choice(TIERS),
                ]
            )
    print(f"customers.csv: {NUMBER_OF_CUSTOMERS:,} rows")


def write_products(rng, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product_id", "product_name", "category", "unit_price"])
        for product_id in range(1, NUMBER_OF_PRODUCTS + 1):
            price = round(rng.uniform(2, 500), 2)
            writer.writerow([product_id, f"Product {product_id}", rng.choice(CATEGORIES), price])
    print(f"products.csv: {NUMBER_OF_PRODUCTS:,} rows")


def write_orders_sample(rng, path):
    """Small CSV with a few deliberately broken rows."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "order_date"]
        )
        for order_id in range(1, 1001):
            moment = datetime(2026, 1, 1) + timedelta(minutes=rng.randint(0, NUMBER_OF_DAYS * 24 * 60))
            quantity = str(rng.randint(1, 5))
            customer_id = str(rng.randint(1, NUMBER_OF_CUSTOMERS))
            # Every 200th row has a bad value, the kind that shows up in real exports.
            if order_id % 200 == 0:
                quantity = "two"
            if order_id % 333 == 0:
                customer_id = "C-" + customer_id
            writer.writerow(
                [
                    order_id,
                    customer_id,
                    rng.randint(1, NUMBER_OF_PRODUCTS),
                    quantity,
                    moment.isoformat(sep=" "),
                    rng.choice(["placed", "shipped", "delivered", "cancelled"]),
                    moment.date().isoformat(),
                ]
            )
    print("orders_sample.csv: 1,000 rows (with a few bad values)")


def write_clicks(rng, folder):
    folder.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    bad_lines = 0
    event_number = 0
    for file_number in range(NUMBER_OF_CLICK_FILES):
        lines = []
        for _ in range(CLICKS_PER_FILE):
            event_number += 1
            moment = datetime(2026, 3, 1) + timedelta(seconds=rng.randint(0, 3 * 24 * 3600))
            product_ids = []
            for _ in range(rng.randint(0, 3)):
                product_ids.append(str(rng.randint(1, NUMBER_OF_PRODUCTS)))
            properties = {
                "device": rng.choice(DEVICES),
                "referrer": rng.choice(["google", "email", "direct", "social"]),
                "product_ids": ",".join(product_ids),
            }
            record = {
                "event_id": f"e{event_number}",
                "customer_id": rng.randint(1, 2000),
                "event_type": rng.choice(EVENT_TYPES),
                "event_ts": moment.isoformat(),
                "page": rng.choice(["/", "/search", "/product", "/cart", "/checkout"]),
                "properties": json.dumps(properties),
            }
            line = json.dumps(record)
            if rng.random() < 0.02:
                # Truncated line: what you get when a log shipper cuts a write short.
                line = line[: len(line) // 2]
                bad_lines += 1
            lines.append(line)
            total_lines += 1
        (folder / f"clicks-{file_number:03d}.json").write_text("\n".join(lines) + "\n")
    print(f"clicks/: {total_lines:,} lines in {NUMBER_OF_CLICK_FILES} files, {bad_lines} malformed")


def write_orders(spark, folder):
    """Big table, built with Spark because Python loops would be slow for 5M rows."""
    orders = (
        spark.range(1, NUMBER_OF_ORDERS + 1, numPartitions=16)
        .withColumnRenamed("id", "order_id")
        .withColumn(
            "customer_id",
            F.when(F.rand(SEED) < HOT_CUSTOMER_SHARE, F.lit(1))
            .otherwise((F.floor(F.rand(SEED + 1) * NUMBER_OF_CUSTOMERS) + 1))
            .cast("long"),
        )
        .withColumn("product_id", (F.floor(F.rand(SEED + 2) * NUMBER_OF_PRODUCTS) + 1).cast("long"))
        .withColumn("quantity", (F.floor(F.rand(SEED + 3) * 5) + 1).cast("int"))
        .withColumn(
            "order_ts",
            F.timestamp_seconds(
                F.lit(int(datetime(2026, 1, 1).timestamp()))
                + F.floor(F.rand(SEED + 4) * NUMBER_OF_DAYS * 24 * 3600)
            ),
        )
        .withColumn(
            "status",
            F.element_at(
                F.array(F.lit("placed"), F.lit("shipped"), F.lit("delivered"), F.lit("cancelled")),
                (F.floor(F.rand(SEED + 5) * 4) + 1).cast("int"),
            ),
        )
        .withColumn("order_date", F.to_date("order_ts"))
    )
    # One file per date partition keeps the folder tidy (90 files, not 90 x 16).
    orders.repartition("order_date").write.mode("overwrite").partitionBy("order_date").parquet(str(folder))
    print(f"orders/: {NUMBER_OF_ORDERS:,} rows, partitioned by order_date")


def main():
    banner("Generating synthetic data into " + str(DATA_DIR))
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    rng = random.Random(SEED)
    write_customers(rng, DATA_DIR / "customers.csv")
    write_products(rng, DATA_DIR / "products.csv")
    write_orders_sample(rng, DATA_DIR / "orders_sample.csv")
    write_clicks(rng, DATA_DIR / "clicks")

    spark = build_spark("generate-data")
    write_orders(spark, DATA_DIR / "orders")
    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
