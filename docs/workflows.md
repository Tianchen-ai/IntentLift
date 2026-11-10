# Recovery and target workflows

## Inspect available workloads

```bash
python tools/scrc_batch.py --list --json
python tools/scrc_batch.py --validate-manifest --json
python tools/scrc_frontend.py --list --json
```

These commands inspect catalog entries and dependency resolution. The catalog is broader than the paper's 24-kernel evaluation cohort; its inventory counts are not acceptance or performance counts.

## Recover and check a source kernel

Configure model access locally. The provider loader accepts `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`, or an explicitly supplied provider JSON file. Keep credentials outside version control. Never attach raw provider configuration or model transcripts to public issues.

For the registered Liger softmax workflow:

```bash
export INTENTIR_DATASET_ROOT=/path/to/Dataset
export INTENTIR_PROVIDER_CONFIG=/secure/path/llm_providers.local.json
export INTENTIR_DEMO_RUN_ROOT=/path/to/evidence/softmax
bash rundemo/03_run_liger_softmax_scrc.sh
```

The script runs fresh Lift, artifact checking, source-bound Replay, Gate 1, and Gate 2. It uses model calls and source-framework compilation. `INTENTIR_PROXY_URL` can be set if the provider requires an HTTP proxy. Run the script with `--help` for all options; do not use the saved public example as fresh recovery evidence.

For other registered Triton cases, inspect `python tools/scrc_frontend.py --help`. CUDA, CuTe, and TileLang have separate compiler/observer paths; use their native entry points and datasets rather than routing them through the Triton workflow. The scripts in `rundemo/` provide catalog and native-frontend diagnostics.

## Compile saved algorithms for targets

The target tools consume a directory of accepted production snapshots. The SwiGLU example's `artifact.json` is a recovery artifact, not a production snapshot directory.

```bash
python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir /path/to/accepted-snapshots \
  --out /path/to/bundles/cuda
python tools/run_cuda_benchmark_bundle.py \
  --bundle /path/to/bundles/cuda \
  --out /path/to/evidence/cuda

python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir /path/to/accepted-snapshots \
  --out /path/to/bundles/rvv
python tools/run_rvv_benchmark_bundle.py \
  --bundle /path/to/bundles/rvv \
  --out /path/to/evidence/rvv
```

The export and target-run steps may execute on different machines. RVV bundles contain compiled target objects; target execution still requires the matching ISA, system libraries, and execution environment. Consult each command's `--help` for toolchain, target and benchmark options.

Keep workload shapes, numerical policies, target configuration, timing regime, and baseline identities with any new result. The frozen paper CSVs in this repository retain their original scope and are not updated by these commands.
