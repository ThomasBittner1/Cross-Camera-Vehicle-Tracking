import cv2
import numpy as np
import time

from config import AppConfig
from tracking import create_tracker_pair, tracks_from_detections
from yolo import load_detection_model


def create_masks(captures, mask_points_pair):
    masks = []
    for camera_index, cap in enumerate(captures):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        points = np.array(mask_points_pair[camera_index], dtype=np.int32)
        cv2.fillPoly(mask, [points], (0, 0, 0))
        masks.append(mask)
    return masks


def draw_tracks(frame, tracks, color):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def draw_detections(frame, detections, color=(0, 255, 255)):
    for detection in detections:
        x1, y1, x2, y2 = detection["bounds"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def draw_frame_count(frame, current_frame_index):
    cv2.putText(
        frame,
        f"Frame {current_frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def run(config=None):
    config = config or AppConfig()
    model = load_detection_model(config.model_path, confidence=0.02, iou=0.7, onnx_input_size=640)
    trackers = create_tracker_pair(config.model_path)

    captures = [cv2.VideoCapture(video_path) for video_path in config.video_paths]
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame_index)

    masks = create_masks(captures, config.mask_points_pair)
    fps = captures[0].get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = 1 #max(1, int(round(1000.0 / fps)))
    paused = False
    current_frame_index = config.start_frame_index
    original_frames = [None for _ in captures]
    measured_fps = 0.0
    previous_frame_time = time.perf_counter()

    for window_name in config.window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret_and_frame_pair = [cap.read() for cap in captures]
            if not all(ret for ret, _ in ret_and_frame_pair):
                break

            frame_pair = [frame for _, frame in ret_and_frame_pair]
            original_frames = [frame.copy() for frame in frame_pair]
            masked_frame_pair = [
                cv2.bitwise_and(frame, mask)
                for frame, mask in zip(frame_pair, masks)
            ]
            if hasattr(model, "predict_many"):
                detection_pair = model.predict_many(masked_frame_pair)
            else:
                detection_pair = [model.predict(frame) for frame in masked_frame_pair]
            tracks_pair = [
                tracks_from_detections(detection_pair[camera_index], trackers[camera_index], original_frames[camera_index])
                for camera_index in range(len(detection_pair))
            ]
            current_frame_time = time.perf_counter()
            elapsed_seconds = current_frame_time - previous_frame_time
            previous_frame_time = current_frame_time
            if elapsed_seconds > 0:
                measured_fps = 1.0 / elapsed_seconds

            for camera_index, tracks in enumerate(tracks_pair):
                draw_frame = original_frames[camera_index].copy()
                draw_detections(draw_frame, detection_pair[camera_index])
                draw_tracks(draw_frame, tracks, config.display.colors_pair[camera_index])
                draw_frame_count(draw_frame, current_frame_index)
                draw_fps(draw_frame, measured_fps)
                cv2.imshow(config.window_names[camera_index], draw_frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

        if not paused:
            current_frame_index += 1

    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
