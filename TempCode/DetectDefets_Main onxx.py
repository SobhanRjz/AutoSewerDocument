import os
import logging
import onnxruntime as ort
import cv2
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import json

@dataclass
class DetectionConfig:
    """Configuration class for defect detection settings"""
    class_names: List[str] = None
    colors: List[List[int]] = None
    score_threshold: float = 0.7
    model_path: str = None
    input_size: tuple = (800, 800)  # Input size for ONNX model inference

def preprocess_image(frame: np.ndarray, input_size: tuple = (400, 400)):
    """Preprocess video frame for ONNX model input."""
    original_frame = frame.copy()
    frame = cv2.resize(frame, input_size)  # Resize to model input size
    frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Convert to channel-first [C, H, W]
    return frame, original_frame


def postprocess_outputs(outputs, original_image, class_names, colors, score_threshold=0.3):
    """Post-process ONNX model outputs and visualize detections."""
    if len(outputs) < 5:
        print("Unexpected ONNX model output format.")
        return original_image

    boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
    if boxes.shape[0] == 0:
        print("No objects detected.")
        return original_image

    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > score_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(class_id)]}: {score:.2f}"
            color = colors[int(class_id)]
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return original_image

class DefectDetectorONNX:
    def __init__(self, config: DetectionConfig):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Initialize ONNX runtime session
        self.session = ort.InferenceSession(config.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        self.class_names = config.class_names
        self.colors = config.colors
        self.score_threshold = config.score_threshold
        self.input_size = config.input_size

    def process_video(self, input_path: str, output_path: str, frame_interval: int = 24):
        """Process video file and detect defects using ONNX model"""
        start_time = time.time()
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            self.logger.error(f"Error opening video file {input_path}")
            return

        writer = self._initialize_video_writer(cap, output_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                preprocessed_frame, original_frame = preprocess_image(frame, self.input_size)
                outputs = self.session.run(self.output_names, {self.input_name: preprocessed_frame})

                annotated_frame = postprocess_outputs(
                    outputs, original_frame, self.class_names, self.colors, self.score_threshold
                )

                writer.write(annotated_frame)

                # Display the frame
                cv2.imshow("Output", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processing complete. Saved to {output_path}")

    def _initialize_video_writer(self, cap: cv2.VideoCapture, output_path: str):
        """Initialize video writer for output"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

def main():
    config = DetectionConfig(
        class_names=["Crack", "Obstacle", "Deposits", "Deformed", "Broken", "Joint Displaced", "Surface Damage", "Root"],
        colors=[
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 128, 128], [128, 0, 128]
        ],
        score_threshold=0.7,
        model_path=r"C:\Users\sobha\Desktop\detectron2\Code\Auto_Sewer_Document\output\model.onnx",
        input_size=(400, 400)
    )

    detector = DefectDetectorONNX(config)

    input_path = r"C:\Users\sobha\Desktop\detectron2\Data\TestFilm\Closed circuit television (CCTV) sewer inspection.mp4"
    output_path = os.path.join(os.getcwd(), os.path.join("output"), os.path.splitext(os.path.basename(input_path))[0] )
    
    detector.process_video(input_path, output_path)

if __name__ == "__main__":
    main()
