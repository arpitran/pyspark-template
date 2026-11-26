import pytest

from pyspark_template.quality import (
    DataQualityError,
    check_columns,
    check_foreign_key,
    check_not_empty,
    check_ratio_at_most,
    run_checks,
)


def results_by_name(results):
    by_name = {}
    for result in results:
        by_name[result["check"]] = result["passed"]
    return by_name


def test_check_columns_finds_duplicates_nulls_and_bad_values(spark):
    df = spark.createDataFrame(
        [(1, "a", "placed"), (1, "b", "shipped"), (2, None, "lost")],
        "id long, name string, status string",
    )
    results = check_columns(df, "id", ["name"], accepted_values={"status": ["placed", "shipped"]})
    assert results_by_name(results) == {
        "unique(id)": False,
        "not_null(name)": False,
        "accepted_values(status)": False,
    }


def test_check_foreign_key_and_not_empty(spark):
    parent = spark.createDataFrame([(1,), (2,)], "id long")
    child = spark.createDataFrame([(1,), (3,)], "id long")
    assert check_foreign_key(child, parent, "id")["passed"] is False
    assert check_not_empty(parent)["passed"] is True
    assert check_not_empty(parent.limit(0))["passed"] is False


def test_run_checks_raises_on_failure():
    with pytest.raises(DataQualityError):
        run_checks([check_ratio_at_most(10, 100, 0.05, "corrupt_ratio")])
    assert run_checks([check_ratio_at_most(1, 100, 0.05, "corrupt_ratio")]) == []
