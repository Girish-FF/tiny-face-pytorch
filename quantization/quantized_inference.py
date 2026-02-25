import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
import onnxruntime as ort
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from layers import PriorBox
from config import get_config
from utils.general import draw_detections, get_output_path
from utils.box_utils import decode, decode_landmarks, nms

# Use Union for Python 3.9 compatibility if needed in this script
from typing import Union, Tuple

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantized ONNX Inference")

    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='quantization/quantized_outputs/retinaface_int8.onnx',
        help='Path to the quantized ONNX model'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='retinaface',
        choices=['retinaface', 'slim', 'rfb'],
        help='Select a model architecture'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='NMS threshold'
    )
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save detection results'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default='assets/test.jpg',
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='quantization/quantized_outputs/inference_results',
        help='Path to the output folder'
    )

    return parser.parse_args()


class FaceONNXInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load ONNX model
        print(f"Loading model from {args.weights}")
        # sess_options = ort.SessionOptions()
        # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(args.weights, providers=['CPUExecutionProvider'])  #, sess_options=sess_options 
        self.cfg = get_config(args.network)
        self.input_name = self.ort_session.get_inputs()[0].name

    def preprocess_image(self, image):
        rgb_mean = np.array([104, 117, 123], dtype=np.float32)
        image = image.astype(np.float32)
        image = cv2.resize(image, (640,360))
        image -= rgb_mean
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        input_type = self.ort_session.get_inputs()[0].type
        if 'float16' in input_type:
            return image.astype(np.float16)
        return image

    def infer(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None
            
        resized_image = cv2.resize(original_image, (640, 360))
        img_height, img_width, _ = resized_image.shape
        image = self.preprocess_image(resized_image)

        # img_height, img_width, _ = original_image.shape
        # image = self.preprocess_image(original_image)

        start = time.time()
        outputs = self.ort_session.run(None, {self.input_name: image})
        inference_time = time.time() - start
        
        loc, conf, landmarks = outputs[0].squeeze(0), outputs[1].squeeze(0), outputs[2].squeeze(0)

        # Generate anchor boxes
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors()

        # Decode boxes and landmarks
        boxes = decode(torch.tensor(loc), priors, self.cfg['variance']).to(self.device)
        landmarks = decode_landmarks(torch.tensor(landmarks), priors, self.cfg['variance']).to(self.device)

        # Scales
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=self.device)
        landmarks = (landmarks * landmark_scale).cpu().numpy()

        scores = conf[:, 1]
        inds = scores > self.args.conf_threshold
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.args.nms_threshold)
        detections, landmarks = detections[keep], landmarks[keep]

        return np.concatenate((detections, landmarks), axis=1), original_image, inference_time

    def run(self):
        if not os.path.exists(self.args.image_path):
            print(f"Path {self.args.image_path} does not exist.")
            return

        image_list = []
        if os.path.isdir(self.args.image_path):
            for root, _, files in os.walk(self.args.image_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_list.append(os.path.join(root, file))
        else:
            if self.args.image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_list.append(self.args.image_path)

        if not image_list:
            print("No images found to process.")
            return

        os.makedirs(self.args.output_path, exist_ok=True)
        results = {}
        total_time = 0
        
        print(f"Processing {len(image_list)} images...")
        for img_path in image_list:
            img_name = os.path.basename(img_path)
            output = self.infer(img_path)
            if output is None: continue
            
            detections, processed_img, inf_time = output
            total_time += inf_time
            
            results[img_name] = {
                "num_detections": len(detections),
                "inference_time": inf_time,
                "detections": detections.tolist()
            }
            
            if self.args.save_image:
                out_path = os.path.join(self.args.output_path, f"res_{img_name}")
                draw_detections(processed_img, detections, self.args.conf_threshold)
                cv2.imwrite(out_path, processed_img)

        # Save results
        with open(os.path.join(self.args.output_path, "results.json"), 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Processed {len(results)} images.")
        print(f"Average inference time: {total_time / len(results):.4f}s")
        print(f"Results saved to {self.args.output_path}")

if __name__ == '__main__':
    args = parse_arguments()
    infer = FaceONNXInference(args)
    infer.run()
