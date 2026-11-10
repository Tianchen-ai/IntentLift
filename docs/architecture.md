# From recovered algorithm to compiler input

IntentLift separates two responsibilities: understanding the optimized source kernel, and implementing the recovered algorithm on a target. The model proposes the former. The compiler takes over after checking.

## The recovered artifact

The serialized artifact has three components:

| Component | Meaning | Code |
| --- | --- | --- |
| `J` | Executable algorithm, tensor signatures, iteration and state, domains, boundary conditions, and effects | `SemanticArchive` in `python/intentir/verify/gate1/model.py` |
| `C` | Concrete source implementation configuration used for recovery checking | `ImplementationConfig` in the same module |
| `pi` | Frozen realization correspondence used to normalize source and Replay observations | `python/intentir/verify/gate1/pi.py` |

`J` is the object reused by the target compiler. Source configuration and correspondence support the recovery check; they do not become a second algorithm implementation for the target. A target may use source configuration as a search prior, while its legal implementation choices are derived from the accepted algorithm and the target's own contracts.

## Two complementary checks

**Gate 1** asks whether a fresh Replay can recover the source's observed structure from the proposed artifact. Source and Replay are independently compiled and observed. A fixed normalization then compares the observations. Replay does not receive source code, source compiler IR, previous Replay programs, or Gate diagnostics. The current implementation supplies source-bound configuration values; the paper protocol distinction is documented in [provenance](provenance.md).

**Gate 2** executes the source, the exact Replay accepted by Gate 1, and a reference implementation of `J`. It compares all three pairs under the workload's input and numerical comparison policy. An acceptance is evidence for those observations and executions, rather than an all-input equivalence proof. Older serialized names such as `SCRC_CERTIFIED` denote this acceptance status.

These checks have different jobs. Structural feedback can identify an incorrect interpretation before target optimization. Execution checks connect the recovered representation to observable results. Neither a well-typed JSON record nor a successful MLIR build alone establishes source equivalence.

## The compiler handoff

`emit_j_module(j, dims, ...)` in [`canonical_j.py`](../python/intentir/frontend/canonical_j.py) materializes Intent MLIR from the algorithm and invocation dimensions. It does not accept source code, Replay, or `pi`.

The C++ dialect is defined in [`include/Intent`](../include/Intent). Its transformations in [`lib/Intent/Conversion`](../lib/Intent/Conversion) lower structured operations and choose physical implementations. The CPU path lowers through Linalg and LLVM; CUDA and RVV paths add their target transformations and search choices. Target compilation and optimization do not invoke a model to rewrite the algorithm.

This is the central reuse boundary: a target can change tiling, vectorization, and memory organization while remaining an implementation of the same accepted algorithm. Correctness still depends on the recovered algorithm and the compiler transformations; repeatable code generation is not itself a proof.

The [SwiGLU example](../examples/swiglu/README.md) makes this boundary concrete: its script passes only the saved `artifact.j` and shape bindings to the compiler, then executes the resulting native code.

## Source frontends and target backends

Triton, CUDA, CuTe, and TileLang retain their own source/compiler adapters and observation logic. The shared Triton runner does not convert every frontend to Triton. Target backends are separate from these source adapters: a CUDA source frontend and a CUDA target backend serve different roles.

Useful entry points:

- `python/intentir/verify/gate1/`: artifact model, Replay, compiler observations, and comparison.
- `python/intentir/frontend/canonical_j.py`: deterministic algorithm materialization.
- `python/intentir/backend/cpu/`: reference lowering, native compilation, and execution.
- `python/intentir/backend/cuda/` and `python/intentir/backend/riscv/`: target compilation and search.
- `python/intentir/ir/production_snapshot.py`: saved algorithm / optional search-prior interface for target use.
