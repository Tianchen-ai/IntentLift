# Compile a recovered SwiGLU algorithm

This example starts after algorithm recovery. It loads the saved `artifact.json`, passes **only its `j` component and shape bindings** into the compiler, emits Intent MLIR, lowers it through Linalg and LLVM, and executes the resulting native code.

For inputs `a`, `b`, and scalar `beta`, the saved algorithm computes:

```text
s = a * beta
out = s * sigmoid(s) * b
```

`run.py` uses float32 arrays of shape `(2, 127)` and `beta = 1`. It compares native execution with this expression using `rtol=1e-5` and `atol=1e-6`. This small executable example needs no source framework, GPU, or model service.

From the repository root, after [building and installing](../../docs/build-and-run.md):

```bash
python examples/swiglu/run.py
```

The generated `output/swiglu/intent.mlir` is suitable for inspection with `build/tools/intent-opt/intent-opt`. `--emit-only` skips native compilation and execution.

## Where the algorithm came from

The artifact is the unchanged second Lift attempt from the real Liger SwiGLU closure preserved in upstream PR #2, commit `bc182c422ebaa0c02563736d123d5406c6144888`. The saved record reports Gate 1 `replay_closed` and Gate 2 passed for its one recorded case. `evidence.json` records those fields and the original receipt hashes.

The saved source-side acceptance and the current example run answer different questions: the former records recovery checking, while the latter demonstrates that the algorithm can be compiled and executed without reading the source or invoking a model. Running this example does not establish a new source-side acceptance or reproduce the paper's full evaluation.
