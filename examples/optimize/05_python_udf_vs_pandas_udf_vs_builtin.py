"""05 - Python UDF vs Arrow UDF vs pandas UDF vs built-in functions, same logic on 5M rows.

Run:   uv run python examples/optimize/05_python_udf_vs_pandas_udf_vs_builtin.py

The logic: label each order as "<STATUS>-<size>", where size is small (1-2), medium (3-4) or large (5).

Look for:
  - Pickled Python UDF (useArrow=False): every row is pickled, sent to a Python worker, and sent back
    (BatchEvalPython in the plan). This was the default before Spark 4.
  - Arrow Python UDF: same function, rows move in Arrow batches (ArrowEvalPython). Spark 4 turns this on
    by default (spark.sql.execution.pythonUDF.arrow.enabled=true), so a plain F.udf(...) gets it.
  - pandas UDF: your function gets a whole pandas Series at a time (ArrowEvalPython). Much less overhead.
  - Built-in: runs in the JVM with code generation. No Python at all. Catalyst can also optimize around it.
"""

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from pyspark_template.session import DATA_DIR, banner, build_spark
from pyspark_template.timing import measure, noop_write, plan_contains, record_benchmark

spark = build_spark("optimize-05-udfs")
orders = spark.read.parquet(str(DATA_DIR / "orders")).select("order_id", "status", "quantity")


def label_one(status, quantity):
    if quantity <= 2:
        size = "small"
    elif quantity <= 4:
        size = "medium"
    else:
        size = "large"
    return status.upper() + "-" + size


pickled_udf = F.udf(label_one, StringType(), useArrow=False)
arrow_udf = F.udf(label_one, StringType())  # Arrow by default on Spark 4


@F.pandas_udf(StringType())
def pandas_label(status: pd.Series, quantity: pd.Series) -> pd.Series:
    size = pd.Series("large", index=quantity.index)
    size = size.mask(quantity <= 4, "medium")
    size = size.mask(quantity <= 2, "small")
    return status.str.upper() + "-" + size


size_column = (
    F.when(F.col("quantity") <= 2, "small").when(F.col("quantity") <= 4, "medium").otherwise("large")
)
builtin_label = F.concat(F.upper("status"), F.lit("-"), size_column)

variants = [
    ("pickled UDF", orders.withColumn("label", pickled_udf("status", "quantity")), "BatchEvalPython"),
    ("arrow UDF", orders.withColumn("label", arrow_udf("status", "quantity")), "ArrowEvalPython"),
    ("pandas UDF", orders.withColumn("label", pandas_label("status", "quantity")), "ArrowEvalPython"),
    ("built-in", orders.withColumn("label", builtin_label), None),
]

banner("All four give the same labels?")
reference = None
for name, df, _ in variants:
    counts = sorted(df.groupBy("label").count().collect())
    if reference is None:
        reference = counts
    print(f"  {name:<11} matches: {counts == reference}")

results = {}
for name, df, operator in variants:
    banner(name)
    if operator is not None:
        print(f"{operator} in plan:", plan_contains(df, operator))
    else:
        print("Python operators in plan:", plan_contains(df, "EvalPython"))
    measure(name, lambda df=df: noop_write(df), results, runs=2)

record_benchmark(
    "05 UDF: pickled Python UDF vs built-in (5M rows)",
    "pickled UDF",
    "built-in",
    results,
    "BatchEvalPython removed; JVM codegen",
)
record_benchmark("05 UDF: arrow vs built-in", "arrow UDF", "built-in", results, "ArrowEvalPython removed")
record_benchmark("05 UDF: pandas vs built-in", "pandas UDF", "built-in", results, "ArrowEvalPython removed")
record_benchmark(
    "05 UDF: pickled vs Arrow (same Python function)",
    "pickled UDF",
    "arrow UDF",
    results,
    "BatchEvalPython -> ArrowEvalPython",
)

print("\nOrder of preference: built-in > pandas UDF / Arrow UDF > pickled Python UDF.")

spark.stop()
