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
