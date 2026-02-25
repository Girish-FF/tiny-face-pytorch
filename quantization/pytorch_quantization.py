import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from models import RetinaFace, SlimFace, RFB
from config import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Quantization')
    parser.add_argument('-w', '--weights', type=str, default='weights/retinaface.pth', help='Path to PyTorch weights')
    parser.add_argument('--network', type=str, default='retinaface', choices=['retinaface', 'slim', 'rfb'], help='Network architecture')
    parser.add_argument('--output_dir', type=str, default='quantization/quantized_outputs', help='Directory to save quantized models')
    parser.add_argument('--calibration_data', type=str, default='test_img', help='Path to calibration images')
    return parser.parse_args()

class QuantizableModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        # Handle tuple output from RetinaFace
        if isinstance(x, tuple):
            return tuple(self.dequant(item) for item in x)
        return self.dequant(x)

def get_calibration_dataloader(image_folder):
    if not os.path.exists(image_folder):
        print(f"Warning: Calibration folder {image_folder} not found.")
        return []
        
    image_list = [
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    
    rgb_mean = np.array([104, 117, 123], dtype=np.float32)
    calibration_data = []
    
    print(f"Loading {len(image_list)} images for calibration...")
    for img_path in image_list:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (640, 640))
        img = img.astype(np.float32)
        img -= rgb_mean
        img = img.transpose(2, 0, 1) # HWC to CHW
        calibration_data.append(torch.from_numpy(img).unsqueeze(0))
    
    return calibration_data

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg = get_config(args.network)
    if cfg is None:
        raise KeyError(f"Config file for {args.network} not found!")

    device = torch.device('cpu') 
    
    # Initialize model
    if args.network == "retinaface":
        base_model = RetinaFace(cfg=cfg)
    elif args.network == "slim":
        base_model = SlimFace(cfg=cfg)
    elif args.network == "rfb":
        base_model = RFB(cfg=cfg)
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} not found.")
        return

    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    
    model_name = os.path.splitext(os.path.basename(args.weights))[0]

    # 1. FP16 Conversion
    print("--- Starting FP16 Half Precision Conversion ---")
    try:
        fp16_model_path = os.path.join(args.output_dir, f"{model_name}_fp16.pth")
        model_fp16 = base_model.half()
        torch.save(model_fp16.state_dict(), fp16_model_path)
        print(f"FP16 model state_dict saved to: {fp16_model_path}")
        base_model.float() # Reset to float32
    except Exception as e:
        print(f"Error during FP16 conversion: {e}")

    # 2. INT8 Static Quantization (Eager Mode)
    print("--- Starting INT8 Static Quantization (Eager Mode) ---")
    try:
        # Use a shallow copy or just wrap the base model
        model_to_quantize = QuantizableModel(base_model)
        model_to_quantize.eval()
        
        # Configure for x86 (fbgemm) or ARM (qnnpack)
        backend = "fbgemm"
        try:
            torch.backends.quantized.engine = backend
        except:
            backend = "qnnpack"
            torch.backends.quantized.engine = backend
            
        model_to_quantize.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        
        # Prepare
        print(f"Preparing model for quantization using {backend} backend...")
        model_prepared = torch.ao.quantization.prepare(model_to_quantize)
        
        # Calibrate
        calib_data = get_calibration_dataloader(args.calibration_data)
        if not calib_data:
            print("No calibration data found. Skipping INT8.")
        else:
            print(f"Calibrating with {len(calib_data)} images...")
            with torch.no_grad():
                for img in calib_data:
                    model_prepared(img)
            
            # Convert
            print("Converting model to quantized version...")
            model_int8 = torch.ao.quantization.convert(model_prepared)
            
            # Save INT8 model
            int8_path = os.path.join(args.output_dir, f"{model_name}_int8.pth")
            torch.save(model_int8.state_dict(), int8_path)
            print(f"INT8 model state_dict saved to: {int8_path}")
            
            # Save as TorchScript (Recommended for quantized PyTorch models)
            example_input = torch.randn(1, 3, 640, 640)
            try:
                traced_model = torch.jit.trace(model_int8, example_input)
                jit_path = os.path.join(args.output_dir, f"{model_name}_int8_jit.pt")
                torch.jit.save(traced_model, jit_path)
                print(f"INT8 JIT model saved to: {jit_path}")
            except Exception as trace_err:
                print(f"Warning: Could not JIT trace the model: {trace_err}")

    except Exception as e:
        print(f"Error during INT8 quantization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
