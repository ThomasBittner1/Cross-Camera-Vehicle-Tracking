from pathlib import Path

import numpy as np
import torch
from boxmot import BotSort


def create_tracker_pair(model_path, frame_rate=30):
    return [
        BotSort(
            reid_weights=Path(model_path),
            device=get_torch_device(),
            half=False,
            with_reid=False,
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            cmc_method=None, #"sof",
            frame_rate=frame_rate,
        )
        for _ in range(2)
    ]


def tracks_from_prediction(result, tracker, frame):
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 8), dtype=np.float32)

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
    clss = result.boxes.cls.cpu().numpy().reshape(-1, 1)
    detections = np.hstack((boxes, confs, clss)).astype(np.float32)
    return tracker.update(detections, frame)


def tracks_from_detections(detections, tracker, frame):
    if not detections:
        return np.empty((0, 8), dtype=np.float32)

    tracker_inputs = [
        [
            *detection["bounds"],
            detection["confidence"],
            -1 if detection["class_id"] is None else detection["class_id"],
        ]
        for detection in detections
    ]
    return tracker.update(np.array(tracker_inputs, dtype=np.float32), frame)


def tracks_from_model(model, frames, trackers, original_frames):
    if hasattr(model, "predict_many"):
        detection_pair = model.predict_many(frames)
    else:
        detection_pair = [model.predict(frame) for frame in frames]
    return [
        tracks_from_detections(detection_pair[camera_index], trackers[camera_index], original_frames[camera_index])
        for camera_index in range(len(detection_pair))
    ]


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
