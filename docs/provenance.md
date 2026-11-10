# Source and evidence provenance

## Source lineage

This public snapshot integrates:

- The original implementation at `zzyu5/Intentlift`, base commit `0878dc065106a0d8c5817d8a75e1c85cf5d75f08`.
- [Upstream PR #2](https://github.com/zzyu5/Intentlift/pull/2), authored by `mortal-two`, head `bc182c422ebaa0c02563736d123d5406c6144888`. It was merged upstream as `322f2c9fab785a647a14547a5a60435099a73321`.
- Existing local RVV / MLIR work on paired rotations, broadcast-preserving fusion, row-vector matmul, row-packet scan, and associated target search / reference support.

The repository starts with a clean publication snapshot. Development histories, local provider configuration, raw model trajectories, private manuscript material, and machine-specific evidence directories are not part of this distribution. Source-file hashes are recorded in `source-manifest.json`.

LLVM/MLIR is an external dependency. The repository includes the Intent dialect and compiler passes written for it. Existing third-party notices remain attached to their respective source.

## Paper protocol and current implementation

The paper results and current integrated code have explicit, separate provenance:

| Aspect | Paper experiment | Current integrated implementation |
| --- | --- | --- |
| Replay configuration input | Configuration slots, with values bound by the external harness | Complete configuration values checked against the source/harness binding |
| Replay source access | Source-isolated | Source-isolated; no source/compiler IR or previous Replay |
| Algorithm materialization | Executable algorithm passed to the target compiler | `emit_j_module(j, dims, ...)`; no source, Replay, or witness input |
| Measurements | Frozen paper experiment records | No new recovery or performance campaign performed for this publication |

In the current code, `_assert_packet_c_matches_artifact` in `python/intentir/verify/gate1/replay.py` enforces the bound configuration. The optional masked arm is a **masked-J diagnostic**, not a claim that configuration values are masked. This implementation update must not be interpreted as an exact rerun of the paper's original Gate 1 protocol.

The executable algorithm remains the target compiler's semantic input. Configuration values used for checking source reconstruction are distinct from a backend's target-specific implementation choices.

## Examples and results

`examples/swiglu/artifact.json` is copied unchanged from the saved, accepted SwiGLU closure in upstream PR #2. `evidence.json` contains selected status fields and hashes of the saved receipts. The example demonstrates recompilation of that artifact on the current compiler; it does not recreate the source-side checks or the paper's full cohort.

`paper/data/` contains existing paper plotting inputs. `paper/data-manifest.json` records original and published hashes. Local filesystem path columns were removed from the mutation table; all other fields are retained. The plotted GPU per-kernel inputs are rounded measurements, while the aggregate CSV preserves the paper's reported geometric means.

This code appendix supplies implementation, a runnable compiler example, and result data. External framework datasets, model services, target hardware, and the full historical raw execution archive are separate inputs to full experimental reproduction.
