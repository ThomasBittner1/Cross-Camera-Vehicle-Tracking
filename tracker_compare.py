import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

from boxmot import BotSort


START_FRAME_INDEX = 950
VIDEO_PATH = r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi"
MODEL_PATH = r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt"
WINDOW_NAME = "tracker_compare_c042"
CONF_THRESHOLD = 0.02
MASK_PTS = [(0, 416), (721, 147), (963, 122), (1074, 197), (244, 959), (1, 955)]


def draw_detection_boxes(frame, result, color, thickness, transparency):
    if result.boxes is None or len(result.boxes) == 0:
        return

    overlay = frame.copy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            overlay,
            f"pred {confs[i]:.2f}",
            (x1, min(frame.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    cv2.addWeighted(overlay, transparency, frame, 1.0 - transparency, 0, frame)


def draw_boxmot_tracks(frame, tracks, color, thickness, transparency):
    overlay = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    cv2.addWeighted(overlay, transparency, frame, 1.0 - transparency, 0, frame)


def draw_ultralytics_tracks(frame, result, color, thickness, transparency):
    if result.boxes is None or len(result.boxes) == 0:
        return

    overlay = frame.copy()
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    ids = result.boxes.id
    track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        track_id = track_ids[i]
        label = f"ultra {track_id}" if track_id is not None else "ultra"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    cv2.addWeighted(overlay, transparency, frame, 1.0 - transparency, 0, frame)



def draw_frame_footer(frame, frame_index, frame_height):
    cv2.putText(
        frame,
        f"Frame {frame_index}",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_predict = YOLO(MODEL_PATH)
    model_track = YOLO(MODEL_PATH)
    tracker = BotSort(
        reid_weights=Path(MODEL_PATH),
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
        frame_rate=30,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_INDEX)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))

    mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    pts = np.array(MASK_PTS, dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    paused = False
    step_one_frame = False
    current_frame_index = START_FRAME_INDEX

    while True:
        should_process_frame = (not paused) or step_one_frame

        if should_process_frame:
            ret, frame = cap.read()
            if not ret:
                break

            masked_frame = cv2.bitwise_and(frame, mask)
            draw_frame = frame.copy()

            pred_result = model_predict.predict(
                source=masked_frame,
                verbose=False,
                conf=CONF_THRESHOLD,
            )[0]

            if pred_result.boxes is not None and len(pred_result.boxes) > 0:
                boxes = pred_result.boxes.xyxy.cpu().numpy()
                confs = pred_result.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = pred_result.boxes.cls.cpu().numpy().reshape(-1, 1)
                detections = np.hstack((boxes, confs, clss)).astype(np.float32)
                boxmot_tracks = tracker.update(detections, frame)
            else:
                boxmot_tracks = np.empty((0, 8), dtype=np.float32)

            ultra_result = model_track.track(
                source=masked_frame,
                verbose=False,
                conf=CONF_THRESHOLD,
                persist=True,
            )[0]

            predicted_ids = (
                pred_result.boxes.cls.int().cpu().tolist()
                if pred_result.boxes is not None and len(pred_result.boxes) > 0
                else []
            )
            boxmot_track_ids = [int(track[4]) for track in boxmot_tracks]
            print(
                f"Frame {current_frame_index}: "
                f"predicted_ids={predicted_ids} "
                f"boxmot_track_ids={boxmot_track_ids}"
            )

            draw_ultralytics_tracks(draw_frame, ultra_result, (255, 0, 0), thickness=8, transparency=0.5)
            draw_detection_boxes(draw_frame, pred_result, (0, 0, 255), thickness=4, transparency=1.0)
            draw_boxmot_tracks(draw_frame, boxmot_tracks, (0, 255, 0), thickness=2, transparency=1.0)
            draw_frame_footer(draw_frame, current_frame_index, frame_height)

            if step_one_frame:
                step_one_frame = False

        cv2.imshow(WINDOW_NAME, draw_frame)
        key = cv2.waitKeyEx(30 if paused else delay_ms)

        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if paused and key == 2555904:
            step_one_frame = True
            continue

        if should_process_frame:
            current_frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
