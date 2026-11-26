# pyspark-template

Runnable PySpark examples, organized by three skills: **writing** Spark code, **debugging** it, and **optimizing** it. Every example runs locally on generated data. Each optimization records before/after timings and shows the change in the query plan that explains them.

Built on PySpark 4.2, Delta Lake 4, Python 3.12.

## What is in here

### Write: `examples/write/`

| # | Example | Techniques |
|---|---|---|
| 01 | `read_formats_and_schemas` | CSV / JSON / Parquet, explicit `StructType` vs `inferSchema`, PERMISSIVE vs FAILFAST, why `count()` hides bad CSV values |
| 02 | `transformations_json_explode` | `from_json`, `explode_outer`, `when`, `regexp_extract`, `inline(map_entries(...))` for JSON maps |
| 03 | `joins_all_types` | inner, left, right, full, `left_semi`, `left_anti` (orphan detection), cross, aliasing |
| 04 | `aggregations_and_pivot` | multi-metric `agg`, `pivot` with explicit values, `rollup`, `approx_count_distinct` |
| 05 | `window_functions` | latest-row dedupe with `row_number`, running totals, `rank` vs `dense_rank`, `lag`, sessionization |
| 06 | `spark_sql_temp_views` | temp views, CTEs, parameterized SQL, proof that SQL and DataFrame code compile to the same plan |
| 07 | `writes_partitioning_modes` | save modes, `partitionBy`, static vs **dynamic partition overwrite**, file-count control |
| 08 | `delta_merge_upsert` | Delta `MERGE` upsert, duplicate-source failure, time travel, schema enforcement / `mergeSchema` |
| 09 | `streaming_watermark_foreachbatch` | Structured Streaming file source, watermark, windows, checkpoint restart, `foreachBatch`, `availableNow` |

### Debug: `examples/debug/`

| # | Example | Techniques |
|---|---|---|
| 01 | `reading_explain_plans` | `explain` modes, reading a physical plan bottom-up, AQE plan before vs after execution, slow-query checklist |
| 02 | `bad_records_quarantine` | corrupt-record column, quarantining bad lines with their source file, failure thresholds, per-file isolation |
| 03 | `common_errors_and_fixes` | `AMBIGUOUS_REFERENCE`, ANSI cast errors vs silent nulls, lazy-evaluation errors, UDF return-type nulls, `maxResultSize` |
| 04 | `spark_ui_and_event_logs` | job descriptions, Spark UI REST API, finding the skewed task by records read, event logs for the History Server |
| 05 | `data_quality_checks` | uniqueness / null / accepted-value / foreign-key checks in one pass, catching a row-multiplying join |

### Optimize: `examples/optimize/`

| # | Example | Before → after |
|---|---|---|
| 01 | `broadcast_join` | `SortMergeJoin` → `BroadcastHashJoin` |
| 02 | `skew_salting_vs_aqe` | skewed sort-merge join → AQE skew join and manual salting |
| 03 | `repartition_coalesce_shuffle_partitions` | 2000 shuffle partitions → 16 and AQE coalescing; `coalesce(1)` vs `repartition(1)` |
| 04 | `cache_persist_when_it_helps` | recompute 3x → cache once; and when caching is pure overhead |
| 05 | `python_udf_vs_pandas_udf_vs_builtin` | Python UDF / Arrow UDF / pandas UDF → built-in functions |
| 06 | `partition_pruning_predicate_pushdown` | filter on a derived date → filter on the partition column; column pruning |
| 07 | `small_files_compaction` | thousands of tiny files → one file per partition |

### Library code and tests

`src/pyspark_template/transforms/` holds pure `DataFrame -> DataFrame` functions (cleaning, enrichment, windows, sessionization). The examples call them and `tests/` covers them with `pyspark.testing.assertDataFrameEqual` and `assertSchemaEqual` on small in-memory DataFrames. I/O lives in the examples, logic lives in the functions, so the logic can be tested without files.

`src/pyspark_template/quality.py` holds data-quality checks used as a pipeline gate.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Java 17+.

```bash
make install        # uv sync
make data           # ~40s: 5M orders (Parquet), 100K customers, 1K products, 50K click events (JSON)
make test           # unit tests
make run-write      # or run-debug, run-optimize, run-all
uv run python examples/optimize/01_broadcast_join.py   # any single example
```

The Makefile looks for Java 17 with `/usr/libexec/java_home`, then Homebrew's `openjdk@17`. Override with `make test JAVA_HOME=/path/to/jdk`.

### The data

`data/generate.py` writes seeded synthetic e-commerce data, so every run produces the same values. Three properties are planted on purpose:
- **Skew:** customer 1 owns about 30% of all orders.
- **Bad values:** a few CSV rows hold `"two"` as a quantity or `"C-123"` as a customer id.
- **Truncated JSON:** about 2% of the click-log lines are cut off.

## Benchmarks

Measured with `make run-optimize`: best of 2-3 runs, local mode, 16-core Apple Silicon laptop. A cluster changes the absolute numbers. The plan changes, and the direction of each result, carry over.

| Example | Before | After | Speedup | What changed in the plan |
|---|---|---|---|---|
| 01 broadcast join (5M orders x 1K products) | sort-merge join: 0.36s | broadcast join: 0.13s | 2.8x | SortMergeJoin + 2 Exchanges -> BroadcastHashJoin, orders not shuffled |
| 02 skew: AQE skew join | plain join: 0.60s | AQE skew join: 0.41s | 1.5x | hot partition split by AQE (SortMergeJoin skew=true) |
| 02 skew: manual salting | plain join: 0.60s | salted join: 0.57s | 1.0x | customer 1 spread over 16 partitions; no gain at this size: customers copied 16x (1.6M rows) |
| 03 shuffle partitions (groupBy 5M rows) | 2000 shuffle partitions: 0.85s | 16 shuffle partitions: 0.16s | 5.4x | 2000 tiny reduce tasks -> 16 right-sized tasks |
| 03 AQE partition coalescing | 2000 shuffle partitions: 0.85s | 2000 + AQE coalesce: 0.24s | 3.5x | AQEShuffleRead coalesced 2000 partitions down to a few |
| 03 one output file | coalesce(1): 5.65s | repartition(1): 4.02s | 1.4x | coalesce collapses the whole stage to 1 task; repartition hashes in parallel, then shuffles |
| 04 cache an expensive, reused result | 3 reports, no cache: 1.41s | 3 reports, cached: 0.63s | 2.2x | window over 5M rows computed once; reports read InMemoryTableScan |
| 04 skip cache for single use | 1 report, cached: 0.85s | 1 report, no cache: 0.07s | 12.0x | no InMemoryRelation: unused sha2 column pruned instead of materialized |
| 05 UDF: pickled Python UDF vs built-in (5M rows) | pickled UDF: 0.53s | built-in: 0.18s | 2.9x | BatchEvalPython removed; JVM codegen |
| 05 UDF: arrow vs built-in | arrow UDF: 0.30s | built-in: 0.18s | 1.7x | ArrowEvalPython removed |
| 05 UDF: pandas vs built-in | pandas UDF: 0.31s | built-in: 0.18s | 1.7x | ArrowEvalPython removed |
| 05 UDF: pickled vs Arrow (same Python function) | pickled UDF: 0.53s | arrow UDF: 0.30s | 1.7x | BatchEvalPython -> ArrowEvalPython |
| 06 partition pruning (1 of 90 days) | filter on to_date(order_ts): 0.10s | filter on order_date (partition): 0.05s | 2.0x | PartitionFilters [] -> [order_date = 2026-03-01]: 1 folder read instead of 90 |
| 06 column pruning (Parquet) | read all columns: 0.04s | read 2 columns: 0.04s | 1.2x | ReadSchema shrinks to customer_id, quantity (within noise at 0.04s) |
| 07 small files (1 month, 1,712,693 rows) | 6,200 small files: 0.94s | 31 compacted files: 0.14s | 6.6x | 200x fewer files: fewer opens, footers and tiny tasks |

Timings under about 0.1s are close to the noise floor. Read those rows for the plan change, not the ratio.

## Lessons learned

_To be written._

## Running the same code on a cluster

Only `session.py` changes between environments:
- **Databricks:** drop the builder. The platform provides `spark`. Package the transforms as a wheel or a Databricks Asset Bundle, and keep the tests running locally.
- **AWS Glue / EMR:** the same functions run inside a Glue job script or a `spark-submit` step.

The local `master("local[*]")` session exists only for development and tests.
