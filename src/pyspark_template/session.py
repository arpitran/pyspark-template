"""One place to build a SparkSession, so every example starts the same way."""

from pathlib import Path

from pyspark.sql import SparkSession

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "generated"
OUTPUT_DIR = REPO_ROOT / "output"
EVENT_LOG_DIR = REPO_ROOT / "spark-events"


def build_spark(app_name, extra_conf=None, with_delta=False, event_log=False):
    """Build a local SparkSession.

    extra_conf: dict of Spark settings that override the defaults below.
    with_delta: add the Delta Lake jars and SQL extensions.
    event_log: write an event log so the Spark History Server can replay the run.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Local default is 200 shuffle partitions: far too many for laptop-sized data.
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
    )

    if event_log:
        EVENT_LOG_DIR.mkdir(exist_ok=True)
        builder = builder.config("spark.eventLog.enabled", "true")
        builder = builder.config("spark.eventLog.dir", str(EVENT_LOG_DIR))

    if extra_conf is not None:
        for key, value in extra_conf.items():
            builder = builder.config(key, value)

    if with_delta:
        from delta import configure_spark_with_delta_pip

        builder = builder.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        builder = builder.config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        builder = configure_spark_with_delta_pip(builder)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def banner(title):
    """Print a section header so the output is easy to follow."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
