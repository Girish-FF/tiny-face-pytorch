import os
import cv2
import json
import time
import torch
import argparse
import numpy as np
import onnxruntime as ort

from layers import PriorBox
from config import get_config
from utils.general import draw_detections, get_output_path
from utils.box_utils import decode, decode_landmarks, nms


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX Inference with Resolution Sweep")

    # Model and device options
    parser.add_argument(
        '-w', '--weights',
        type=str,
        default='./weights/retinaface_mv2.onnx',
        help='Path to the trained model weights'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='retinaface',
        choices=['retinaface', 'slim', 'rfb'],
        help='Select a model architecture for face detection'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.02,
        help='Confidence threshold for filtering detections (kept low to capture weak detections)'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=5000,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.4,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=750,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=0.6,
        help='Visualization threshold for displaying detections'
    )

    # Image input
    parser.add_argument(
        '--image-path',
        type=str,
        default='./assets/test.jpg',
        help='Path to the input image or folder'
    )

    # Output folder path
    parser.add_argument(
        '--output-path',
        type=str,
        default='output',
        help='Path to the output folder'
    )

    # Resolution sweep settings
    parser.add_argument(
        '--start-width',
        type=int,
        default=1920,
        help='Starting width for resolution sweep (height auto-computed to maintain 16:9)'
    )
    parser.add_argument(
        '--start-height',
        type=int,
        default=1080,
        help='Starting height for resolution sweep'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=160,
        help='Pixel decrement per step (applied to width; height scaled proportionally)'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=160,
        help='Minimum width to stop the sweep'
    )

    # Confidence threshold for "reliable" detection tracking
    parser.add_argument(
        '--high-conf-threshold',
        type=float,
        default=0.75,
        help='High confidence threshold for tracking smallest reliably detected face'
    )

    return parser.parse_args()


def get_face_area(detection):
    """Return the pixel area of a detection box [x1, y1, x2, y2, score]."""
    x1, y1, x2, y2 = detection[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def get_face_dimensions(detection):
    """Return (width, height) of a detection box."""
    x1, y1, x2, y2 = detection[:4]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


class FaceONNXInference:
    def __init__(
        self,
        model_path,
        model_name,
        conf_threshold=0.02,
        pre_nms_topk=5000,
        nms_threshold=0.4,
        post_nms_topk=750,
        vis_threshold=0.6
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pre_nms_topk = pre_nms_topk
        self.nms_threshold = nms_threshold
        self.post_nms_topk = post_nms_topk
        self.vis_threshold = vis_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ONNX model
        self.ort_session = ort.InferenceSession(model_path)
        print("ONNX providers:", self.ort_session.get_providers())

        # Config for prior boxes
        self.cfg = get_config(model_name)

    def preprocess_image(self, image):
        rgb_mean = np.array([104, 117, 123], dtype=np.float32)
        image = image.astype(np.float32)
        image -= rgb_mean
        image = image.transpose(2, 0, 1)       # HWC → CHW
        image = np.expand_dims(image, axis=0)  # → (1, C, H, W)

        input_type = self.ort_session.get_inputs()[0].type
        if 'float16' in input_type:
            return image.astype(np.float16)
        return image

    def infer(self, image):
        """
        Run inference on a pre-loaded (and possibly resized) BGR image array.
        Returns (detections, time_taken) where detections is None if no face found.
        """
        img_height, img_width, _ = image.shape
        preprocessed = self.preprocess_image(image)

        start = time.time()
        outputs = self.ort_session.run(None, {'input': preprocessed})
        time_taken = time.time() - start

        loc, conf, landmarks = (
            outputs[0].squeeze(0),
            outputs[1].squeeze(0),
            outputs[2].squeeze(0),
        )

        # Generate anchor boxes
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors()

        # Decode boxes and landmarks
        boxes = decode(torch.tensor(loc), priors, self.cfg['variance']).to(self.device)
        landmarks_decoded = decode_landmarks(
            torch.tensor(landmarks), priors, self.cfg['variance']
        ).to(self.device)

        bbox_scale = torch.tensor([img_width, img_height] * 2, device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=self.device)
        landmarks_decoded = (landmarks_decoded * landmark_scale).cpu().numpy()

        scores = conf[:, 1]  # confidence for class "face"

        # Filter by confidence threshold
        inds = scores > self.conf_threshold
        boxes, landmarks_decoded, scores = (
            boxes[inds], landmarks_decoded[inds], scores[inds]
        )

        if len(boxes) == 0:
            return None, time_taken

        # Sort by score descending
        order = scores.argsort()[::-1][:self.pre_nms_topk]
        boxes, landmarks_decoded, scores = (
            boxes[order], landmarks_decoded[order], scores[order]
        )

        # NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, self.nms_threshold)
        detections = detections[keep]
        landmarks_decoded = landmarks_decoded[keep]

        # Keep top-k
        detections = detections[:self.post_nms_topk]
        landmarks_decoded = landmarks_decoded[:self.post_nms_topk]

        return np.concatenate((detections, landmarks_decoded), axis=1), time_taken

    def run_resolution_sweep(
        self,
        image_path,
        output_root,
        start_width=1920,
        start_height=1080,
        step_size=160,
        min_width=160,
        high_conf_threshold=0.75,
    ):
        """
        Sweep from (start_width × start_height) down to min_width, decrementing by
        step_size each iteration (aspect ratio kept constant).

        For every resolution:
          - Saves annotated image  →  <output_root>/<base_name>/<WxH>/<base_name>_<WxH>.jpg
          - Saves detection JSON   →  <output_root>/<base_name>/<WxH>/detections.json

        After the sweep a summary JSON is written to:
          <output_root>/<base_name>/sweep_summary.json
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_root = os.path.join(output_root, base_name)
        os.makedirs(image_output_root, exist_ok=True)

        aspect_ratio = start_height / start_width  # e.g. 0.5625 for 16:9

        # ── tracking variables ──────────────────────────────────────────────
        sweep_summary = []

        # smallest face at any confidence above self.conf_threshold
        global_smallest_area   = float('inf')
        global_smallest_info   = None  # (resolution_str, detection_dict)

        # smallest face above high_conf_threshold
        hc_smallest_area       = float('inf')
        hc_smallest_info       = None

        last_resolution_with_detection = None

        # ── sweep ───────────────────────────────────────────────────────────
        width = start_width
        while width >= min_width:
            height = int(round(width * aspect_ratio))
            res_str = f"{width}x{height}"

            resized = cv2.resize(original_image, (width, height))
            detections, time_taken = self.infer(resized)

            res_dir = os.path.join(image_output_root, res_str)
            os.makedirs(res_dir, exist_ok=True)

            # ── build per-detection records ──────────────────────────────
            detection_records = []
            if detections is not None and len(detections) > 0:
                last_resolution_with_detection = res_str
                for det in detections:
                    x1, y1, x2, y2, score = det[0], det[1], det[2], det[3], det[4]
                    w  = max(0.0, x2 - x1)
                    h  = max(0.0, y2 - y1)
                    area = w * h
                    lm   = det[5:].tolist()  # 10 landmark coords

                    record = {
                        "bbox":       [float(x1), float(y1), float(x2), float(y2)],
                        "width_px":   float(w),
                        "height_px":  float(h),
                        "area_px2":   float(area),
                        "confidence": float(score),
                        "landmarks":  [float(v) for v in lm],
                    }
                    detection_records.append(record)

                    # update global smallest (any confidence)
                    if area < global_smallest_area:
                        global_smallest_area = area
                        global_smallest_info = (res_str, record)

                    # update high-confidence smallest
                    if score >= high_conf_threshold and area < hc_smallest_area:
                        hc_smallest_area = area
                        hc_smallest_info = (res_str, record)

            num_detections      = len(detection_records)
            num_above_high_conf = sum(
                1 for r in detection_records if r["confidence"] >= high_conf_threshold
            )

            # ── save face crops ───────────────────────────────────────────
            crops_dir = os.path.join(res_dir, "crops")
            if detection_records:
                os.makedirs(crops_dir, exist_ok=True)
            for idx, (det, record) in enumerate(zip(detections, detection_records)):
                x1, y1, x2, y2 = det[:4]
                # clamp to image bounds
                cx1 = max(0, int(x1))
                cy1 = max(0, int(y1))
                cx2 = min(resized.shape[1], int(x2))
                cy2 = min(resized.shape[0], int(y2))
                crop = resized[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    record["crop_path"] = None
                    continue
                crop_name = f"face_{idx:03d}_conf{record['confidence']:.3f}.jpg"
                crop_path = os.path.join(crops_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                # store relative path for portability
                record["crop_path"] = os.path.join("crops", crop_name)

            # ── save annotated visualisation ─────────────────────────────
            vis_image = resized.copy()
            if detections is not None:
                draw_detections(vis_image, detections, self.vis_threshold)
            vis_path = os.path.join(res_dir, f"{base_name}_{res_str}.jpg")
            cv2.imwrite(vis_path, vis_image)

            # ── save per-resolution JSON ─────────────────────────────────
            res_json = {
                "image":              image_path,
                "resolution":         res_str,
                "width":              width,
                "height":             height,
                "inference_time_sec": float(time_taken),
                "total_detections":   num_detections,
                "detections_above_high_conf_threshold": num_above_high_conf,
                "high_conf_threshold": high_conf_threshold,
                "detections":         detection_records,
            }
            json_path = os.path.join(res_dir, "detections.json")
            with open(json_path, 'w') as f:
                json.dump(res_json, f, indent=4)

            print(
                f"[{res_str}] faces={num_detections:3d}  "
                f"(conf≥{high_conf_threshold}: {num_above_high_conf})  "
                f"time={time_taken*1000:.1f}ms"
            )

            # ── summarise this step ───────────────────────────────────────
            sweep_summary.append({
                "resolution":        res_str,
                "width":             width,
                "height":            height,
                "total_detections":  num_detections,
                "detections_above_high_conf": num_above_high_conf,
                "inference_time_sec": float(time_taken),
            })

            # ── stop early if zero detections ─────────────────────────────
            if num_detections == 0:
                print(f"  → No detections at {res_str}. Stopping sweep.")
                break

            width -= step_size

        # ── write sweep summary JSON ─────────────────────────────────────────
        summary = {
            "image": image_path,
            "sweep_step_px": step_size,
            "conf_threshold": self.conf_threshold,
            "high_conf_threshold": high_conf_threshold,
            "last_resolution_with_detection": last_resolution_with_detection,
            "smallest_face_any_conf": {
                "resolution":  global_smallest_info[0] if global_smallest_info else None,
                "face_width_px":  global_smallest_info[1]["width_px"]  if global_smallest_info else None,
                "face_height_px": global_smallest_info[1]["height_px"] if global_smallest_info else None,
                "area_px2":       global_smallest_info[1]["area_px2"]  if global_smallest_info else None,
                "confidence":     global_smallest_info[1]["confidence"] if global_smallest_info else None,
                "bbox":           global_smallest_info[1]["bbox"]       if global_smallest_info else None,
            },
            "smallest_face_high_conf": {
                "resolution":     hc_smallest_info[0] if hc_smallest_info else None,
                "face_width_px":  hc_smallest_info[1]["width_px"]  if hc_smallest_info else None,
                "face_height_px": hc_smallest_info[1]["height_px"] if hc_smallest_info else None,
                "area_px2":       hc_smallest_info[1]["area_px2"]  if hc_smallest_info else None,
                "confidence":     hc_smallest_info[1]["confidence"] if hc_smallest_info else None,
                "bbox":           hc_smallest_info[1]["bbox"]       if hc_smallest_info else None,
            },
            "per_resolution": sweep_summary,
        }

        summary_path = os.path.join(image_output_root, "sweep_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 60)
        print(f"Sweep complete for: {image_path}")
        print(f"Last resolution with detection : {last_resolution_with_detection}")
        if global_smallest_info:
            r = global_smallest_info[1]
            print(
                f"Smallest detectable face       : {r['width_px']:.1f}×{r['height_px']:.1f} px  "
                f"(area={r['area_px2']:.1f})  conf={r['confidence']:.3f}  "
                f"@ {global_smallest_info[0]}"
            )
        if hc_smallest_info:
            r = hc_smallest_info[1]
            print(
                f"Smallest face (conf≥{high_conf_threshold})     : {r['width_px']:.1f}×{r['height_px']:.1f} px  "
                f"(area={r['area_px2']:.1f})  conf={r['confidence']:.3f}  "
                f"@ {hc_smallest_info[0]}"
            )
        else:
            print(f"Smallest face (conf≥{high_conf_threshold})     : no detection met this threshold")
        print(f"Summary saved → {summary_path}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_arguments()

    inferencer = FaceONNXInference(
        model_path=args.weights,
        model_name=args.network,
        conf_threshold=args.conf_threshold,
        pre_nms_topk=args.pre_nms_topk,
        nms_threshold=args.nms_threshold,
        post_nms_topk=args.post_nms_topk,
        vis_threshold=args.vis_threshold,
    )

    # Collect all images to process
    image_files = []
    if os.path.isfile(args.image_path):
        image_files = [args.image_path]
    elif os.path.isdir(args.image_path):
        for root, _, files in os.walk(args.image_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, f))
    else:
        print(f"[ERROR] Input path does not exist: {args.image_path}")
        exit(1)

    if not image_files:
        print("[ERROR] No valid image files found.")
        exit(1)

    os.makedirs(args.output_path, exist_ok=True)

    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        inferencer.run_resolution_sweep(
            image_path=img_path,
            output_root=args.output_path,
            start_width=args.start_width,
            start_height=args.start_height,
            step_size=args.step_size,
            min_width=args.min_width,
            high_conf_threshold=args.high_conf_threshold,
        )