"""
Post-training quantization script using OpenVINO + NNCF.

Usage examples:
    python -m src.calibration.quantize_openvino \
        --model models/models/cnn_feature_extractor_4ch_val.onnx \
        --calib-dir src/calibration/hailo_calibration/input_0 \
        --output-dir models/ir_int8 \
        --limit 128 \
        --preset performance

Assumptions:
    - Single-input model.
    - NPY files already match the model's expected layout/normalization.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from nncf import Dataset, quantize, QuantizationPreset
from openvino.runtime import Core
import openvino
import openvino.runtime as ov
import sys
import types

# Workaround: some OpenVINO builds do not export Node at top-level, but NNCF expects it
if not hasattr(openvino, "Node") and hasattr(ov, "Node"):
    openvino.Node = ov.Node  # type: ignore[attr-defined]
# Export Output as well for NNCF compatibility
if not hasattr(openvino, "Output") and hasattr(ov, "Output"):
    openvino.Output = ov.Output  # type: ignore[attr-defined]
# Export Input for NNCF compatibility
if not hasattr(openvino, "Input") and hasattr(ov, "Input"):
    openvino.Input = ov.Input  # type: ignore[attr-defined]
# Some OpenVINO builds do not expose `openvino.op`; NNCF expects it
if not hasattr(openvino, "op") and hasattr(ov, "op"):
    openvino.op = ov.op  # type: ignore[attr-defined]
if "openvino.op" not in sys.modules and hasattr(ov, "op"):
    sys.modules["openvino.op"] = ov.op  # ensure import openvino.op works
# Export opsets explicitly for NNCF (expects openvino.opset13, etc.)
for opset_name in ("opset13", "opset12", "opset11", "opset10"):
    if hasattr(ov, opset_name) and f"openvino.{opset_name}" not in sys.modules:
        sys.modules[f"openvino.{opset_name}"] = getattr(ov, opset_name)
# Provide openvino.utils.node_factory for NNCF
if hasattr(ov, "opset_utils"):
    if not hasattr(openvino, "utils"):
        openvino.utils = types.SimpleNamespace()  # type: ignore[attr-defined]
    nf_mod = types.ModuleType("openvino.utils.node_factory")
    nf_mod.NodeFactory = ov.opset_utils.NodeFactory  # type: ignore[attr-defined]
    sys.modules["openvino.utils.node_factory"] = nf_mod
    sys.modules.setdefault("openvino.utils", nf_mod)


class NpyDataset(Dataset):
    """Wrap a list of npy paths to feed NNCF."""

    def __init__(self, paths: Iterable[Path]):
        self.paths = list(paths)
        # For NNCF compatibility: wrap indices as data source
        self._data_source = list(range(len(self.paths)))
        self._transform_func = lambda idx: self[idx]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        arr = np.load(self.paths[idx])
        arr = np.asarray(arr, dtype=np.float32)

        # Normalize to match runtime preprocessing (RGB+edge)
        # 1) Scale to 0-1
        arr = arr / 255.0
        # 2) Ensure NCHW
        if arr.ndim == 3:
            if arr.shape[-1] in (1, 3, 4):  # HWC -> CHW
                arr = arr.transpose(2, 0, 1)
            arr = arr[None, ...]  # add batch
        elif arr.ndim == 4:
            if arr.shape[-1] in (1, 3, 4) and arr.shape[1] not in (1, 3, 4):
                arr = arr.transpose(0, 3, 1, 2)

        # 3) Per-channel normalize (RGB + edge)
        mean = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225, 0.5], dtype=np.float32)
        if arr.shape[1] == 4:
            arr = (arr - mean[None, :, None, None]) / std[None, :, None, None]
        else:
            # Fallback: only RGB available
            arr = (arr - mean[: arr.shape[1]][None, :, None, None]) / std[: arr.shape[1]][
                None, :, None, None
            ]

        # Return as tuple to match single-input signature
        return (arr,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenVINO INT8 PTQ using NNCF")
    parser.add_argument("--model", required=True, help="FP32 ONNX model path")
    parser.add_argument(
        "--calib-dir",
        required=True,
        help="Directory containing calibration *.npy files",
    )
    parser.add_argument(
        "--output-dir",
        default="models/ir_int8",
        help="Where to save the INT8 IR (model.xml/bin)",
    )
    parser.add_argument(
        "--preset",
        default="performance",
        choices=["performance", "mixed"],
        help="NNCF quantization preset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N calibration samples (0 = all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    calib_dir = Path(args.calib_dir)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not calib_dir.exists():
        raise FileNotFoundError(calib_dir)

    paths = sorted(Path(p) for p in glob.glob(str(calib_dir / "*.npy")))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No npy files found under {calib_dir}")

    dataset = NpyDataset(paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Model: {model_path}")
    print(f"[+] Calibration samples: {len(paths)} from {calib_dir}")
    print(f"[+] Preset: {args.preset}")
    print(f"[+] Saving to: {output_dir}")

    # Explicitly read ONNX into OpenVINO IR to avoid backend inference issues
    core = Core()
    ov_model = core.read_model(model_path)

    preset = QuantizationPreset.PERFORMANCE if args.preset == "performance" else QuantizationPreset.MIXED
    compressed_model = quantize(ov_model, dataset, preset=preset)
    xml_path = output_dir / "model.xml"
    bin_path = output_dir / "model.bin"
    try:
        compressed_model.save_model(output_dir)  # type: ignore[attr-defined]
    except AttributeError:
        ov.serialize(compressed_model, xml_path, bin_path)
    print(f"[+] Done. INT8 saved to {xml_path} / {bin_path}")


if __name__ == "__main__":
    main()
