from pathlib import Path

import numpy as np
import torch
from boxmot import BotSort


def create_tracker_pair(model_path, device, frame_rate=30):
    return [
        BotSort(
            reid_weights=Path(model_path),
            device=device,
            half=False,
            with_reid=False,
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            cmc_method="sof",
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


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
