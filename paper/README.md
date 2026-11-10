# Existing paper results

These tables are frozen inputs used by the paper's plots. No recovery or performance campaign was rerun to populate this repository. `data-manifest.json` provides source-relative paths, hashes, row counts, and any removed columns.

## GPU performance

`gpu_24_per_kernel.csv` contains the same 24 kernels on RTX 5090 D and H100 80GB HBM3. `gpu_24_aggregate.csv` preserves the reported geometric means: **1.305×** and **1.440×** over the original Triton implementations, respectively. IntentLift is no slower than the source in 23 of 24 cases on each device.

Original field names are retained for traceability:

| Field | Meaning |
| --- | --- |
| `*_source_over_full` | Original Triton latency / IntentLift latency |
| `*_g_schedule` | Fixed-configuration latency / schedule-search latency |
| `*_g_lowering` | Schedule-search latency / structurally optimized latency |

`full` in these historical CSV fields means the IntentLift configuration with both schedule search and structural lowering. Per-kernel values are rounded, so recomputing a geometric mean from them may differ slightly from the original aggregate.

```bash
python -m pip install matplotlib
python paper/plot_results.py
```

This regenerates `figures/gpu_speedups.png` and `.svg` directly from the CSV. It does not run kernels or modify the data.

## SG2044 RVV attention forward

`rvv_flash_attention_forward.csv` describes a single workload: `B=1, Q=128, H=1, D=64, K=128`, mixed f16/f32, one thread, cold-data timing with a 128 MiB working set. The target ISA in the record is `rv64gcv_zfh_zvfh-vlen128`.

| Implementation | Latency (µs) |
| --- | ---: |
| Fixed configuration | 4039.1085 |
| Schedule search | 3831.638 |
| IntentLift structural lowering | 1234.536 |
| LLVM RVV C baseline | 4425.76 |

The resulting speedups are **3.104×** over schedule search and **3.585×** over the LLVM RVV C baseline. These are workload-specific numbers, not geometric means over 24 RVV kernels.

## Algorithm-mutation study

`j_mutations.csv` contains 300 existing cases. A mutation is accepted only when Gate 1 and all three Gate 2 pairwise comparisons pass. Applying that rule to the table gives:

| Mutation group | Cases | Rejected | Accepted |
| --- | ---: | ---: | ---: |
| Activated by the evaluation inputs | 250 | 250 | 0 |
| Not activated by the evaluation inputs | 50 | 46 | 4 |
| Total | 300 | 296 | 4 |

The four accepted mutations were not distinguished by the recorded observations and inputs. These results characterize the bounded checks; they are not a formal all-input equivalence claim. The published CSV removes only two local filesystem path columns (`artifact`, `output_dir`); measurements and outcomes are unchanged.

## Frontend extension cost

`frontend_extension_cost.csv` separates extension work from shared infrastructure. For 12 groups at equal detection, marginal code is **112 SLoC** for IntentLift versus **1,243 SLoC** for the hybrid translation-validation implementation. The shared 623-SLoC frontend observation layer is excluded from both marginal counts. The full from-zero footprints in the original accounting are 2,311 and 3,018 SLoC, respectively; the marginal comparison is not a whole-system size comparison.

## Relationship to the code

The integrated source includes implementation updates after these measurements. See [source provenance](../docs/provenance.md) for the Replay protocol distinction and source lineage. This appendix provides these result tables and the current implementation without relabeling old timings as measurements of the current checkout.
