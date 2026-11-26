import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """One small local session shared by every test. Starting the JVM is the slow part."""
    session = (
        SparkSession.builder.master("local[2]")
        .appName("pyspark-template-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
