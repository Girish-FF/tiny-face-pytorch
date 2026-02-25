"""
verify_onnx.py
Quick sanity check for an exported ONNX model.
Validates the graph, checks opset, and runs a dummy inference pass.

Usage:
  python verify_onnx.py --model retinaface_op13.onnx
  python verify_onnx.py --model retinaface_op13.onnx --input_size 640 640
"""

import argparse
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Verify an ONNX model export.")
    parser.add_argument("--model", "-m", required=True, help="Path to .onnx file")
    parser.add_argument(
        "--input_size", nargs=2, type=int, default=[640, 640],
        metavar=("W", "H"),
        help="Dummy input resolution for inference test (W H). Default: 640 640"
    )
    return parser.parse_args()


def check_onnx_graph(model_path: str):
    """Load and validate the ONNX graph structure."""
    import onnx
    print("[1] Loading and validating ONNX graph ...")
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
        print("    Graph validation : PASSED ✓")
    except onnx.checker.ValidationError as e:
        print(f"    Graph validation : FAILED ✗\n    {e}")
        sys.exit(1)

    opsets = {op.domain: op.version for op in model.opset_import}
    main_opset = opsets.get("", opsets.get("ai.onnx", "unknown"))
    print(f"    Opset version    : {main_opset}")

    # Check for prim::Constant nodes -- should be zero in the final ONNX graph
    prim_nodes = [n for n in model.graph.node if "prim" in n.op_type.lower()]
    if prim_nodes:
        print(f"    prim:: nodes     : {len(prim_nodes)} found (may cause issues)")
    else:
        print("    prim:: nodes     : none (export folded them correctly ✓)")

    return model, main_opset


def check_io_shapes(model_path: str):
    """Print input/output names and shapes."""
    import onnxruntime
    print("\n[2] Inspecting model inputs / outputs ...")
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inputs, outputs = sess.get_inputs(), sess.get_outputs()

    for inp in inputs:
        dynamic = any(isinstance(d, str) or d is None for d in inp.shape)
        tag = "dynamic" if dynamic else "static"
        print(f"    Input  '{inp.name}' : shape={inp.shape} dtype={inp.type} [{tag}]")
    for out in outputs:
        print(f"    Output '{out.name}' : shape={out.shape} dtype={out.type}")
    return sess, inputs


def run_dummy_inference(sess, inputs, input_size):
    """Feed a random tensor through the model and confirm it runs without error."""
    import onnxruntime
    print("\n[3] Running dummy inference ...")
    w, h = input_size
    # Build dummy input matching the model's expected dtype
    inp = inputs[0]
    dtype = np.float32 if "float" in inp.type else np.uint8
    dummy = np.random.randn(1, 3, h, w).astype(dtype)
    print(f"    Input tensor shape : {dummy.shape}  dtype={dummy.dtype}")

    try:
        outputs = sess.run(None, {inp.name: dummy})
        print(f"    Inference          : PASSED ✓")
        for i, out in enumerate(outputs):
            print(f"    Output[{i}] shape   : {out.shape}  dtype={out.dtype}")
    except Exception as e:
        print(f"    Inference          : FAILED ✗\n    {e}")
        sys.exit(1)


def report_model_size(model_path: str):
    import os
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"\n[4] Model size : {size_mb:.2f} MB")


def main():
    args = parse_args()
    model, opset = check_onnx_graph(args.model)
    sess, inputs = check_io_shapes(args.model)
    run_dummy_inference(sess, inputs, args.input_size)
    report_model_size(args.model)

    print("\n" + "=" * 50)
    if opset >= 13:
        print("  ✓ Model is ready for QDQ + per-channel INT8 quantization.")
        print("  Run: python onnx_int8_quantization.py --input <model> \\")
        print("           --mode static --calibration_data <images/>")
    else:
        print(f"  ⚠ Opset {opset}: quantization will auto-use QOperator + per-channel.")
    print("=" * 50)


if __name__ == "__main__":
    main()