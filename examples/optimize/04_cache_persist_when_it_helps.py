"""04 - cache()/persist(): helps when an expensive result is reused, hurts when it is not.

Run:   uv run python examples/optimize/04_cache_persist_when_it_helps.py

Look for:
  - Without cache, every action recomputes the whole lineage. Here that is a window over 5M rows
    (latest order per customer), three times.
  - cache() is lazy. The first action fills the cache; later actions read it (InMemoryTableScan).
  - The best cache candidates are expensive to compute and small to keep: 5M rows in, 100K rows out.
  - Caching a wide DataFrame you use once is slower: the cache materializes EVERY column, including
    ones the query would otherwise have pruned away.
  - unpersist() when done. Cached data holds memory that joins and aggregations need.
"""

from pyspark import StorageLevel
from pyspark.sql import functions as F

from pyspark_template import schemas
from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import measure, plan_contains, record_benchmark
from pyspark_template.transforms.enrich import add_order_revenue
from pyspark_template.transforms.windows import latest_order_per_customer

spark = build_spark("optimize-04-cache")
orders = spark.read.parquet(str(DATA_DIR / "orders"))
products = spark.read.option("header", True).schema(schemas.PRODUCTS).csv(str(DATA_DIR / "products.csv"))


def last_orders():
    """Expensive (shuffle + sort of 5M rows) but small (one row per customer)."""
    return add_order_revenue(latest_order_per_customer(orders), products)


def three_reports(df):
    """Three questions about customers' most recent orders, like a daily reporting job."""
    df.groupBy("category").agg(F.sum("revenue")).collect()
    df.groupBy("status").agg(F.count("*")).collect()
    df.groupBy(F.date_trunc("week", "order_ts")).agg(F.avg("revenue")).collect()


results = {}

banner("BEFORE: no cache - the 5M-row window runs once per report")


def uncached_run():
    three_reports(last_orders())


measure("3 reports, no cache", uncached_run, results)

banner("AFTER: cache the 100K-row result once, then three reports")


def cached_run():
    cached = last_orders().cache()
    cached.count()  # an action fills the cache
    three_reports(cached)
    cached.unpersist()


measure("3 reports, cached", cached_run, results)
record_benchmark(
    "04 cache an expensive, reused result",
    "3 reports, no cache",
    "3 reports, cached",
    results,
    "window over 5M rows computed once; reports read InMemoryTableScan",
)

banner("What the plan looks like when reading from cache")
cached = last_orders().cache()
cached.count()
report = cached.groupBy("category").agg(F.sum("revenue"))
print("InMemoryTableScan in plan:", plan_contains(report, "InMemoryTableScan"))
print("Storage level:", cached.storageLevel)
cached.unpersist()

banner("When cache hurts: a wide DataFrame used once")


def wide_orders():
    return orders.withColumn("fingerprint", F.sha2(F.concat_ws("|", "order_id", "status"), 256))


single_use = {}
measure("1 report, no cache", lambda: wide_orders().groupBy("status").count().collect(), single_use)


def single_cached():
    df = wide_orders().cache()
    df.groupBy("status").count().collect()
    df.unpersist()


measure("1 report, cached", single_cached, single_use)
extra = single_use["1 report, cached"] - single_use["1 report, no cache"]
print(f"Cache cost {extra:+.2f}s. Without cache, Spark prunes the unused fingerprint column and never")
print("computes sha2. With cache, it hashes all 5M rows to store a column nobody reads.")
record_benchmark(
    "04 skip cache for single use",
    "1 report, cached",
    "1 report, no cache",
    single_use,
    "no InMemoryRelation: unused sha2 column pruned instead of materialized",
)

banner("persist() levels")
print("cache() = persist(MEMORY_AND_DISK) for DataFrames: partitions that don't fit in memory go to disk.")
print("DISK_ONLY: keep memory for the work itself when recomputing is very expensive.")
print("MEMORY_AND_DISK_2: two copies, survives losing an executor, costs twice the memory.")
disk_only = last_orders().persist(StorageLevel.DISK_ONLY)
disk_only.count()
print("DISK_ONLY storage level:", disk_only.storageLevel)
disk_only.unpersist()

spark.stop()
