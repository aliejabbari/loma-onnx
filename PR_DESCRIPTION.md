# PR: ONNX export + C++ / Jetson / DGX-Spark deployment

**Title:** `Add ONNX export, a C++ inference library, and Jetson/Spark deployment + benchmarks`

---

## Summary
This PR makes LoMa deployable outside Python without touching the PyTorch model code.
It adds:

- **ONNX export** of every variant (B, B128, R, L, G) — detector, descriptor, and
  matcher exported as separate self-contained graphs, each validated against PyTorch.
- **Resolution / keypoint presets** (`fast`, `balanced`, `quality`, `wide`, …) tuned for
  edge devices (Jetson Orin Nano 8 GB).
- **A reusable C++ library** (`cpp/`) on ONNX Runtime (CUDA EP → CPU fallback) with a
  3-line API, OpenCV preprocessing, and `find_package(LoMa)` integration.
- **A benchmark + chart suite** and a from-source `onnxruntime-gpu` build script for the
  DGX Spark GB10 (sm_121), where no prebuilt wheel exists.

Nothing in `src/loma/` is modified — the export wrappers live in standalone scripts.

## Why
Local feature matchers are most useful inside SfM / visual-localization / robotics
pipelines, which are frequently C++ and run on edge GPUs. ONNX + a thin C++ wrapper makes
LoMa a drop-in there, and the presets keep it real-time on 8 GB Jetsons.

## What's added
```
export_onnx.py        detector/descriptor/matcher → ONNX (+ validation)
export_jetson.py      resolution/keypoint presets
compare_onnx.py       end-to-end ONNX-vs-PyTorch comparison
bench_sweep.py        GPU benchmark sweep + charts
benchmark_*.py        latency references
build_ort_gpu.sh      build onnxruntime-gpu for GB10 / sm_121
cpp/                  C++ library + CMake + examples (see cpp/README.md)
ONNX_DEPLOY.md        full deployment guide
docs/                 benchmark.json + charts
```

## Validation (ONNX reproduces PyTorch)
| stage | metric | result |
|-------|--------|--------|
| detector | keypoint-set agreement / probs | 99.4–99.85% / ~5e-8 |
| descriptor (B) | cosine similarity (mean/min) | 1.00000 / 1.0000 |
| matcher | match-index agreement | 100% (B/R/L/G), 99.95% (B128) |
| end-to-end (B128) | matches & ≤1 px overlap | 984 vs 984, 99.5% |

## Benchmark (NVIDIA GB10, onnxruntime CUDA EP, B128)
`fast` 175 ms / 5.7 FPS · `wide` 283 ms / 1093 matches · `quality` 544 ms / 944 matches.
Full table + charts in `ONNX_DEPLOY.md`.

## How to test
```bash
uv sync && uv pip install onnxruntime onnxscript
uv run export_onnx.py --variants B128
uv run compare_onnx.py          # expects ~99.5% match overlap vs PyTorch
```

## Implementation notes
- Uses the **dynamo (torch.export) ONNX exporter**; the legacy exporter emits
  ORT-invalid graphs (bad `Concat` axis, `MaxPool` dilations).
- DINOv2's `@torch.compiler.disable` + internal `inference_mode` are stripped for export;
  the detector's torchvision `Normalize` (data-dependent branch) is replaced with plain
  arithmetic so `torch.export` can trace it.
- Detector exports at a fixed resolution + keypoint count (baked into the graph); the
  matcher is fully dynamic.

## Checklist
- [x] No changes to `src/loma/` model code
- [x] Exports validated numerically against PyTorch
- [x] C++ library compiles against ONNX Runtime headers
- [x] Large `.onnx` files git-ignored (ship via Releases/LFS)
- [x] Docs + benchmarks included
