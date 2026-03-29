import cv2
import numpy as np
from ultralytics import YOLO

from boxmot import OcSort


START_FRAME_INDEX = 950
VIDEO_PATH = r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi"
MODEL_PATH = r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt"
WINDOW_NAME_OCSORT = "tracker_compare_c042_ocsort"
WINDOW_NAME_ULTRA = "tracker_compare_c042_ultralytics"
CONF_THRESHOLD = 0.02
MASK_PTS = [(0, 416), (721, 147), (963, 122), (1074, 197), (244, 959), (1, 955)]

PRED_COLOR = (255, 0, 0)
TRACK_COLOR = (0, 0, 255)
ULTRA_TRACK_COLOR = (0, 255, 0)
PRED_THICKNESS = 4


def draw_detection_boxes(frame, result, color):
    if result.boxes is None or len(result.boxes) == 0:
        return

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, PRED_THICKNESS)
        cv2.putText(
            frame,
            f"pred {confs[i]:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_ocsort_tracks(frame, tracks, color):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ocsort {track_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_ultralytics_tracks(frame, result, color):
    if result.boxes is None or len(result.boxes) == 0:
        return

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    ids = result.boxes.id
    track_ids = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        track_id = track_ids[i]
        label = f"ultra {track_id}" if track_id is not None else "ultra"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_legend(frame):
    entries = [
        ("predict", PRED_COLOR),
        ("ocsort", TRACK_COLOR),
        ("ultralytics track", ULTRA_TRACK_COLOR),
    ]
    y = 30
    for text, color in entries:
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28


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
    model_predict = YOLO(MODEL_PATH)
    model_track = YOLO(MODEL_PATH)
    tracker = OcSort()

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_INDEX)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))

    mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    pts = np.array(MASK_PTS, dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    cv2.namedWindow(WINDOW_NAME_OCSORT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_NAME_ULTRA, cv2.WINDOW_NORMAL)
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
            draw_frame_ocsort = frame.copy()
            draw_frame_ultra = frame.copy()

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
                ocsort_tracks = tracker.update(detections, frame)
            else:
                ocsort_tracks = np.empty((0, 8), dtype=np.float32)

            ultra_result = model_track.track(
                source=masked_frame,
                verbose=False,
                conf=CONF_THRESHOLD,
                persist=True,
            )[0]

            draw_detection_boxes(draw_frame_ocsort, pred_result, PRED_COLOR)
            draw_ocsort_tracks(draw_frame_ocsort, ocsort_tracks, TRACK_COLOR)
            draw_legend(draw_frame_ocsort)
            draw_frame_footer(draw_frame_ocsort, current_frame_index, frame_height)

            draw_ultralytics_tracks(draw_frame_ultra, ultra_result, ULTRA_TRACK_COLOR)
            draw_legend(draw_frame_ultra)
            draw_frame_footer(draw_frame_ultra, current_frame_index, frame_height)

            if step_one_frame:
                step_one_frame = False

        cv2.imshow(WINDOW_NAME_OCSORT, draw_frame_ocsort)
        cv2.imshow(WINDOW_NAME_ULTRA, draw_frame_ultra)
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
