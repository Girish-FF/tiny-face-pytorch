"""
onnx_int8_quantization.py
─────────────────────────
Convert any ONNX model to INT8 precision using ONNXRuntime's quantization API.
Supports models with DYNAMIC input shapes (e.g. face detectors that accept any
image size at runtime). Calibration images are resized to --calib_size just for
the calibration pass; the quantized model retains the original dynamic axes.

Supports two modes
  --mode dynamic   No calibration data required. Quantizes weights only;
                   activations are quantized at runtime. Fastest to run.
  --mode static    Requires a folder of representative images (--calibration_data).
                   Quantizes both weights AND activations offline for best accuracy.

Usage examples
  # Dynamic (no images needed)
  python onnx_int8_quantization.py --input weights/retinaface.onnx

  # Static -- dynamic-input model, images resized to 640x640 for calibration
  python onnx_int8_quantization.py --input weights/retinaface.onnx \
      --mode static --calibration_data path/to/images/

  # Override the calibration resolution (width height)
  python onnx_int8_quantization.py --input weights/retinaface.onnx \
      --mode static --calibration_data path/to/images/ --calib_size 1280 720

  # Override output path
  python onnx_int8_quantization.py --input model.onnx --output weights/model_int8.onnx

Requirements
  pip install onnx onnxruntime opencv-python numpy
  (opencv-python is only needed for static mode)
"""

import os
import sys
import argparse
import numpy as np


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to INT8 using ONNXRuntime quantization."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input float32 ONNX model."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for the output INT8 ONNX model. "
             "Defaults to <input_stem>_int8.onnx in the same directory."
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic", "static"],
        default="static",
        help="Quantization mode. 'static' is more accurate (default). "
             "'dynamic' needs no calibration data."
    )
    parser.add_argument(
        "--calibration_data",
        default=None,
        help="[static mode] Folder of representative images (.jpg/.png/.jpeg). "
             "50-200 images is typically sufficient."
    )
    parser.add_argument(
        "--calib_size",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("WIDTH", "HEIGHT"),
        help="[static mode] Resolution to resize calibration images to (W H). "
             "For dynamic-input models this only affects calibration, not inference. "
             "Default: 640 640. Tip: use a size typical of your real inputs."
    )
    parser.add_argument(
        "--rgb_mean",
        nargs=3,
        type=float,
        default=[104.0, 117.0, 123.0],
        metavar=("B", "G", "R"),
        help="[static mode] Per-channel mean (BGR order) to subtract during "
             "preprocessing. Default: 104 117 123 (RetinaFace / VGG convention)."
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        default=True,
        help="Per-channel weight quantization (better accuracy). Default: True."
    )
    parser.add_argument(
        "--reduce_range",
        action="store_true",
        default=False,
        help="Use 7-bit range instead of 8-bit to avoid overflow on some CPUs. "
             "Enable this if you see accuracy degradation after quantization."
    )
    parser.add_argument(
        "--quant_format",
        choices=["QDQ", "QOperator"],
        default="QDQ",
        help="[static mode] Quantization format. "
             "QDQ (default): inserts QuantizeLinear/DeQuantizeLinear nodes which "
             "ONNXRuntime fuses into single quantized kernels on x86/x64 -- fastest "
             "on modern CPUs. "
             "QOperator: replaces ops with quantized variants directly -- better for "
             "ARM or custom runtimes that do not support QDQ fusion."
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Calibration data reader (static mode only)
# ------------------------------------------------------------------------------

def _build_calibration_reader(image_folder, input_name, calib_size, rgb_mean):
    """
    Returns a CalibrationDataReader that streams preprocessed images.

    For dynamic-input models we pick a single fixed resolution (calib_size) for
    the calibration pass only. This gives the quantizer representative activation
    ranges without restricting the model to that size at inference time.
    """
    try:
        import cv2
    except ImportError:
        sys.exit(
            "ERROR: opencv-python is required for static quantization.\n"
            "Install with:  pip install opencv-python"
        )

    from onnxruntime.quantization import CalibrationDataReader

    class _ImageReader(CalibrationDataReader):
        def __init__(self):
            self._mean = np.array(rgb_mean, dtype=np.float32)   # BGR, shape (3,)
            self._size = tuple(calib_size)                       # (W, H) for cv2

            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            self._paths = [
                os.path.join(image_folder, f)
                for f in sorted(os.listdir(image_folder))
                if f.lower().endswith(exts)
            ]
            if not self._paths:
                sys.exit(
                    f"ERROR: No images found in '{image_folder}'.\n"
                    "Supported formats: jpg, jpeg, png, bmp, webp."
                )
            print(f"  Found {len(self._paths)} calibration images.")
            print(f"  Resizing each to {calib_size[0]}x{calib_size[1]} for calibration "
                  f"(model input axes remain dynamic).")
            self._iter = None

        def _preprocess(self, path):
            img = cv2.imread(path)
            if img is None:
                print(f"  Warning: could not read '{path}', skipping.")
                return None

            # Preserve aspect ratio with letterboxing so face proportions are
            # realistic, which gives better calibration statistics than a plain
            # stretch resize for a face detection model.
            img = _letterbox(img, self._size)

            img = img.astype(np.float32)
            img -= self._mean           # BGR mean subtraction
            img = img.transpose(2, 0, 1)            # HWC -> CHW
            img = np.expand_dims(img, axis=0)       # add batch dim  (1, 3, H, W)
            return {input_name: img}

        def get_next(self):
            if self._iter is None:
                self._iter = (self._preprocess(p) for p in self._paths)
            for sample in self._iter:
                if sample is not None:
                    return sample
            return None

        def rewind(self):
            self._iter = None

    return _ImageReader()


def _letterbox(img, target_wh):
    """
    Resize img to fit inside target_wh (W, H) while preserving aspect ratio.
    Pads with the dataset mean colour (grey) to fill the remaining space.
    This keeps face proportions realistic for calibration.
    """
    import cv2
    tw, th = target_wh
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)  # neutral grey pad
    pad_top  = (th - nh) // 2
    pad_left = (tw - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = img
    return canvas


# ------------------------------------------------------------------------------
# Pre-processing helper (shape inference + constant folding)
# ------------------------------------------------------------------------------

def _preprocess_model(input_path: str) -> str:
    """
    Run ONNXRuntime's quant_pre_process (shape inference, constant folding).
    For dynamic-input models we pass skip_symbolic_shape=True so the dynamic
    axes are preserved rather than being collapsed to fixed values.
    Returns the path to the pre-processed model.
    """
    pre_path = input_path + ".preprocessed.onnx"
    print("  Running shape inference and constant folding ...")
    try:
        from onnxruntime.quantization import shape_inference
        shape_inference.quant_pre_process(
            input_model_path=input_path,
            output_model_path=pre_path,
            # skip_symbolic_shape=True preserves dynamic axes (e.g. batch, H, W)
            # instead of trying to resolve them to concrete values.
            skip_symbolic_shape=True,
        )
    except Exception as exc:
        print(f"  Warning: pre-processing failed ({exc}). Proceeding without it.")
        return input_path
    return pre_path


# ------------------------------------------------------------------------------
# Quantization routines
# ------------------------------------------------------------------------------

def run_dynamic_quantization(input_path: str, output_path: str, args) -> None:
    """
    Dynamic INT8 quantization.
    Weights are quantized at export time; activations are quantized at runtime.
    No calibration data needed. Works well with dynamic input shapes.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n[Dynamic INT8 Quantization]")
    pre_path = _preprocess_model(input_path)

    print("  Quantizing ...")
    quantize_dynamic(
        model_input=pre_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )

    if pre_path != input_path and os.path.exists(pre_path):
        os.remove(pre_path)


def run_static_quantization(input_path: str, output_path: str, args) -> None:
    """
    Static INT8 quantization with QDQ format.
    Collects activation range statistics from calibration images, then folds
    them into the graph. Gives the best runtime performance.

    Dynamic axes are preserved: the quantized model still accepts any image
    size at inference time -- only the calibration pass uses a fixed size.
    """
    import onnxruntime
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        QuantFormat,
        CalibrationMethod,
    )

    if not args.calibration_data:
        sys.exit(
            "ERROR: --calibration_data is required for static quantization.\n"
            "Provide a folder of representative images, or switch to --mode dynamic."
        )
    if not os.path.isdir(args.calibration_data):
        sys.exit(
            f"ERROR: calibration_data path does not exist: '{args.calibration_data}'"
        )

    print("\n[Static INT8 Quantization — dynamic input axes preserved]")
    pre_path = _preprocess_model(input_path)

    # Auto-detect the model's input tensor name
    print("  Detecting model input name ...")
    sess = onnxruntime.InferenceSession(pre_path, providers=["CPUExecutionProvider"])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape   # e.g. [1, 3, 'height', 'width'] for dynamic
    print(f"  Input name  : '{input_name}'")
    print(f"  Input shape : {input_shape}  "
          f"({'dynamic' if any(isinstance(d, str) or d is None for d in input_shape) else 'static'})")
    del sess  # release file handle before quantization opens it

    reader = _build_calibration_reader(
        image_folder=args.calibration_data,
        input_name=input_name,
        calib_size=args.calib_size,
        rgb_mean=args.rgb_mean,
    )

    print("  Collecting calibration statistics and quantizing ...")
    # QDQ: Q/DQ nodes are fused by ONNXRuntime into quantized kernels at
    #      runtime on x86/x64 -- no actual extra computation despite the
    #      extra nodes visible in the graph. Best for modern CPUs.
    # QOperator: replaces ops in-place. Better for ARM / custom runtimes.
    fmt = QuantFormat.QDQ if args.quant_format == "QDQ" else QuantFormat.QOperator
    quantize_static(
        model_input=pre_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=fmt,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        # MinMax is safest for face detection (avoids clipping rare bright pixels).
        # Switch to CalibrationMethod.Percentile if activations look saturated.
        calibrate_method=CalibrationMethod.MinMax,
    )

    if pre_path != input_path and os.path.exists(pre_path):
        os.remove(pre_path)


# ------------------------------------------------------------------------------
# Verification + size reporting
# ------------------------------------------------------------------------------

def _verify_model(model_path: str) -> None:
    """Load the quantized model and print its I/O spec as a sanity check."""
    import onnxruntime
    print("\n[Verifying output model]")
    sess = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for inp in sess.get_inputs():
        print(f"  Input : '{inp.name}'  shape={inp.shape}  dtype={inp.type}")
    for out in sess.get_outputs():
        print(f"  Output: '{out.name}'  shape={out.shape}  dtype={out.type}")
    print("  Model loaded successfully ✓")


def _report_sizes(original: str, quantized: str) -> None:
    orig_mb  = os.path.getsize(original)  / 1e6
    quant_mb = os.path.getsize(quantized) / 1e6
    ratio    = orig_mb / quant_mb if quant_mb else float("inf")
    print(f"\n[Size comparison]")
    print(f"  Original  : {orig_mb:.1f} MB")
    print(f"  INT8      : {quant_mb:.1f} MB  ({ratio:.1f}x smaller)")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input model not found: '{args.input}'")

    # Resolve output path
    output_path = args.output or (os.path.splitext(args.input)[0] + "_int8.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 60)
    print(f"  Input model  : {args.input}")
    print(f"  Output model : {output_path}")
    print(f"  Mode         : {args.mode}")
    if args.mode == "static":
        print(f"  Calib folder : {args.calibration_data}")
        print(f"  Calib size   : {args.calib_size[0]}x{args.calib_size[1]} (W x H)")
        print(f"  BGR mean     : {args.rgb_mean}")
        print(f"  Quant format : {args.quant_format}")
    print(f"  Per-channel  : {args.per_channel}")
    print(f"  Reduce range : {args.reduce_range}")
    print("=" * 60)

    if args.mode == "dynamic":
        run_dynamic_quantization(args.input, output_path, args)
    else:
        run_static_quantization(args.input, output_path, args)

    if os.path.exists(output_path):
        _verify_model(output_path)
        _report_sizes(args.input, output_path)
        print(f"\nDone! INT8 model saved to: {output_path}")
    else:
        sys.exit("ERROR: Quantization finished but output file was not created.")


if __name__ == "__main__":
    main()