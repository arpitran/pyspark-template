"""Timing and plan helpers for the optimize examples."""

import time
from contextlib import contextmanager

from pyspark_template.session import REPO_ROOT

BENCHMARKS_FILE = REPO_ROOT / "results" / "benchmarks.md"


@contextmanager
def time_it(label, results):
    """Time a block and store seconds in results[label].

    Spark is lazy: only an action (count, write, collect) runs the work.
    So the timed block must end with an action, or you are timing nothing.
    """
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    results[label] = seconds
    print(f"[time] {label}: {seconds:.2f}s")


def noop_write(df):
    """Run the whole query without keeping the output.

    The "noop" format executes every stage, then throws the rows away. It is a
    fair way to time a query: count() lets Spark skip columns it does not need.
    """
    df.write.format("noop").mode("overwrite").save()


def plan_text(df):
    """Return the formatted physical plan as a string (explain() only prints)."""
    return df._jdf.queryExecution().explainString(
        df.sparkSession._jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted")
    )


def final_plan_text(df):
    """Physical plan after the query ran. With AQE on, this is the plan Spark actually used.

    Call it after an action on the same DataFrame. Before any action, AQE plans
    show "isFinalPlan=false" and do not include runtime changes such as a join
    switched to broadcast or a skewed partition split.
    """
    return df._jdf.queryExecution().executedPlan().toString()


def plan_contains(df, operator_name, use_final_plan=False):
    text = final_plan_text(df) if use_final_plan else plan_text(df)
    return operator_name in text


def record_benchmark(example, before_label, after_label, results, plan_change):
    """Append one row to results/benchmarks.md and print the speedup."""
    before = results[before_label]
    after = results[after_label]
    speedup = before / after if after > 0 else float("inf")
    print(f"[result] {before_label} {before:.2f}s -> {after_label} {after:.2f}s ({speedup:.1f}x)")

    BENCHMARKS_FILE.parent.mkdir(exist_ok=True)
    if not BENCHMARKS_FILE.exists():
        header = "| Example | Before | After | Speedup | What changed in the plan |\n|---|---|---|---|---|\n"
        BENCHMARKS_FILE.write_text(header)
    row = (
        f"| {example} | {before_label}: {before:.2f}s | {after_label}: {after:.2f}s "
        f"| {speedup:.1f}x | {plan_change} |\n"
    )
    with open(BENCHMARKS_FILE, "a") as handle:
        handle.write(row)
