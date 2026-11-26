"""One place to build a SparkSession, so every example starts the same way."""

import re
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
        # local mode runs everything in the driver JVM. 4g hit intermittent OutOfMemoryError
        # with 16 cores reading Parquet at once; 6g leaves headroom.
        .config("spark.driver.memory", "6g")
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


def short_error(error, max_length=220):
    """The useful line of a Spark error: the one with the error class, like [MALFORMED_RECORD_IN_PARSING...].

    A Py4J error's first line is often just "An error occurred while calling o63.collectToPython".
    """
    lines = str(error).splitlines()
    error_class = re.compile(r"\[[A-Z][A-Z0-9_.]+\]")
    class_lines = []
    for line in lines:
        if error_class.search(line):
            class_lines.append(line[error_class.search(line).start() :].strip())
    if class_lines:
        # Errors wrap each other (FAILED_READ_FILE wrapping MALFORMED_RECORD_IN_PARSING).
        # Show the outer one and the root cause.
        outer = class_lines[0]
        root = class_lines[-1]
        if root == outer:
            return outer[:max_length]
        return outer[:100] + " ... caused by " + root[:max_length]
    for line in lines:
        if "Exception:" in line:
            return line.strip()[:max_length]
    return lines[0][:max_length] if lines else repr(error)


def banner(title):
    """Print a section header so the output is easy to follow."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
