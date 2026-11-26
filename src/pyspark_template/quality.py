"""Small data-quality checks that run in as few passes over the data as possible.

Each check returns a dict: {"check": name, "passed": bool, "detail": str}.
run_checks prints a report and raises if any check failed, so a pipeline
stops before it writes bad data downstream.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityError(Exception):
    pass


def check_not_empty(df: DataFrame, name="not_empty"):
    # limit(1) stops after the first row instead of counting everything.
    has_rows = len(df.limit(1).collect()) == 1
    return {"check": name, "passed": has_rows, "detail": "has rows" if has_rows else "no rows"}


def check_columns(df: DataFrame, key_column, not_null_columns, accepted_values=None):
    """Uniqueness, null counts and accepted values, all computed in ONE aggregation pass."""
    aggregations = [F.count("*").alias("total_rows"), F.countDistinct(key_column).alias("distinct_keys")]
    for column_name in not_null_columns:
        aggregations.append(F.sum(F.col(column_name).isNull().cast("int")).alias(f"nulls__{column_name}"))
    if accepted_values is not None:
        for column_name, allowed in accepted_values.items():
            is_bad = ~F.col(column_name).isin(allowed) & F.col(column_name).isNotNull()
            aggregations.append(F.sum(is_bad.cast("int")).alias(f"bad_values__{column_name}"))

    stats = df.agg(*aggregations).collect()[0].asDict()

    results = []
    duplicates = stats["total_rows"] - stats["distinct_keys"]
    results.append(
        {
            "check": f"unique({key_column})",
            "passed": duplicates == 0,
            "detail": f"{duplicates} duplicate keys",
        }
    )
    for column_name in not_null_columns:
        null_count = stats[f"nulls__{column_name}"] or 0
        results.append(
            {"check": f"not_null({column_name})", "passed": null_count == 0, "detail": f"{null_count} nulls"}
        )
    if accepted_values is not None:
        for column_name in accepted_values:
            bad_count = stats[f"bad_values__{column_name}"] or 0
            results.append(
                {
                    "check": f"accepted_values({column_name})",
                    "passed": bad_count == 0,
                    "detail": f"{bad_count} unexpected values",
                }
            )
    return results


def check_foreign_key(child: DataFrame, parent: DataFrame, key_column, name=None):
    """Every child key must exist in the parent. left_anti returns the orphans."""
    orphans = (
        child.select(key_column).distinct().join(parent.select(key_column), on=key_column, how="left_anti")
    )
    orphan_count = orphans.count()
    return {
        "check": name or f"foreign_key({key_column})",
        "passed": orphan_count == 0,
        "detail": f"{orphan_count} orphan keys",
    }


def check_ratio_at_most(bad_rows, total_rows, max_ratio, name):
    ratio = bad_rows / total_rows if total_rows else 0.0
    return {
        "check": name,
        "passed": ratio <= max_ratio,
        "detail": f"{bad_rows}/{total_rows} = {ratio:.2%} (limit {max_ratio:.0%})",
    }


def run_checks(results, raise_on_failure=True):
    failed = []
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['check']}: {result['detail']}")
        if not result["passed"]:
            failed.append(result["check"])
    if failed and raise_on_failure:
        raise DataQualityError("Failed checks: " + ", ".join(failed))
    return failed
