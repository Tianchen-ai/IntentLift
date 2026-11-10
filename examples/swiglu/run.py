#!/usr/bin/env python3
"""Compile a saved recovered algorithm; no model calls or timing campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from intentir.backend.cpu.pipeline import execute_on_cpu_outputs
from intentir.frontend.canonical_j import emit_j_module
from intentir.verify.gate1.model import Gate1Artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/swiglu"))
    args = parser.parse_args()
    artifact = Gate1Artifact.from_json(
        json.loads(Path(__file__).with_name("artifact.json").read_text())
    )
    dims = {"x0": 2, "x1": 127}
    args.out.mkdir(parents=True, exist_ok=True)
    module_path = args.out / "intent.mlir"
    module_path.write_text(emit_j_module(artifact.j, dims))
    print(f"Intent MLIR: {module_path}")
    if args.emit_only:
        return

    rng = np.random.default_rng(17)
    a = rng.normal(size=(2, 127)).astype(np.float32)
    b = rng.normal(size=(2, 127)).astype(np.float32)
    beta = np.array(1.0, dtype=np.float32)
    result = execute_on_cpu_outputs(artifact.j, dims, [a, b, beta])
    scaled = a * beta
    expected = scaled / (1.0 + np.exp(-scaled)) * b
    actual = result.outputs["t3"]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    error = float(np.max(np.abs(actual - expected)))
    print("Native CPU compilation and numerical comparison: PASS")
    print(f"Shape: 2 x 127; dtype: float32; maximum absolute error: {error:.3g}")
    print("This recompiles a saved J; it does not rerun Lift, Replay, or Gate acceptance.")


if __name__ == "__main__":
    main()
