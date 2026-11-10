# Build and run

The public example uses Linux, Python 3.10 or later, NumPy, and LLVM/MLIR 20.1.8. Install CMake, Ninja, Clang, the LLVM/MLIR development packages and tools, and the MLIR C runner library. Distribution package names vary; LLVM and MLIR must come from the same installation.

Run the build commands in the [root README](../README.md#quick-start). Python dependencies for the saved example are installed with `python -m pip install -e '.[verify]'`.

For a non-system LLVM installation, set both CMake package directories and the runtime tool paths:

```bash
export INTENT_OPT="$PWD/build/tools/intent-opt/intent-opt"
export MLIR_OPT=/path/to/llvm/bin/mlir-opt
export MLIR_TRANSLATE=/path/to/llvm/bin/mlir-translate
export LLVM_LIB=/path/to/llvm/lib
export CLANG=/path/to/llvm/bin/clang
```

`LLVM_LIB` must contain `libmlir_c_runner_utils.so`. These variables are resolved in `python/intentir/backend/toolchain/tools.py`.

## Run the saved algorithm

```bash
python examples/swiglu/run.py
```

The script materializes Intent MLIR, compiles a native shared object through the CPU backend, executes it in an isolated process, and checks the output. It uses a saved algorithm and does not call an LLM, recover a new kernel, or measure target performance.

To inspect only materialization, without the native toolchain:

```bash
python examples/swiglu/run.py --emit-only
```

To render the original paper figures for the README:

```bash
python -m pip install pymupdf
python paper/render_figures.py
```

## Fresh recovery and target execution

Fresh Lift/Replay requires the optional model worker dependencies:

```bash
python -m pip install -e '.[verify,lift-agent]'
```

It also needs the selected source framework, its compiler, workload source and harness, and a compatible GPU execution environment. The source-framework dependencies are separate from the minimal compiler example. Native CUDA, CuTe, and TileLang workflows need their corresponding framework installations and external source datasets.

RVV execution additionally requires a RISC-V toolchain/sysroot and an RVV machine. Paper RVV results were measured on SG2044; building on an x86 host alone does not reproduce those timings. See [workflows](workflows.md) for target bundle entry points.
