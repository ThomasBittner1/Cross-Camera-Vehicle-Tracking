import cv2
import numpy as np
import time

import embedding_utils
import geometry_utils
from config import AppConfig
from cross_camera_matcher import CrossCameraMatcher
from tracking import create_trackers_by_camera, predict_and_track
from visualization import Visualizer
from yolo import load_detection_model


def _create_masks(captures, mask_points_by_camera):
    masks = []
    for camera_index, cap in enumerate(captures):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        pts = np.array(mask_points_by_camera[camera_index], dtype=np.int32)
        cv2.fillPoly(mask, [pts], (0, 0, 0))
        masks.append(mask)
    return masks



def _register_mouse_callbacks(config, pending_click_by_camera):
    for camera_index, window_name in enumerate(config.window_names):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param, frame_idx=camera_index):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_click_by_camera[frame_idx] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)


def _handle_pending_clicks(pending_click_by_camera, isolated_track_id_by_camera, frame_draw_data_by_camera):
    for camera_index in [0, 1]:
        pending_click = pending_click_by_camera[camera_index]
        if pending_click is None:
            continue

        clicked_box = None
        for box in reversed(frame_draw_data_by_camera[camera_index]["boxes"]):
            if geometry_utils.point_inside_box(pending_click, box["coords"]):
                clicked_box = box
                break

        if clicked_box is not None:
            isolated_track_id_by_camera[camera_index] = clicked_box["track_id"]
        elif isolated_track_id_by_camera[camera_index] is not None:
            isolated_track_id_by_camera[camera_index] = None

        pending_click_by_camera[camera_index] = None


def _track_crossed_line(track, previous_centers, crossing_line):
    track_id = int(track[4])
    x1, y1, x2, y2 = map(int, track[:4])
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    previous_center = previous_centers.get(track_id)
    previous_centers[track_id] = center
    if previous_center is None:
        return False

    return geometry_utils.segments_intersect(
        previous_center,
        center,
        crossing_line[0],
        crossing_line[1],
    )


def _process_track(camera_index, track, tracks, original_frame, current_frame_index, delay_ms, config, cross_camera_matcher, previous_centers_by_camera, crossed_times_by_camera):
    track_id = int(track[4])
    x1, y1, x2, y2 = map(int, track[:4])
    label = f"ID {track_id}"
    is_good_crop = False
    one_or_more_cars_just_crossed = False

    if _track_crossed_line(track, previous_centers_by_camera[camera_index], config.cross_lines[camera_index]):
        if track_id not in crossed_times_by_camera[camera_index]:
            one_or_more_cars_just_crossed = True
            crossed_times_by_camera[camera_index][track_id] = current_frame_index * (delay_ms / 1000.0)
            if camera_index == 0:
                cross_camera_matcher.record_camera_0_crossing(track_id)

    if track_id in crossed_times_by_camera[camera_index]:
        label = f"{label} crossed"

    if camera_index == 0:
        is_good_crop = cross_camera_matcher.record_camera_0_crop(track, tracks, original_frame)
    elif camera_index == 1:
        cross_camera_matcher.update_camera_1_matches(track_id, crossed_times_by_camera)
        cross_camera_matcher.update_elapsed_times(track_id, crossed_times_by_camera)
    else:
        raise ValueError(f"Unknown camera index: {camera_index}")

    return {
        "track_id": track_id,
        "coords": (x1, y1, x2, y2),
        "label": label,
        "label_color": [255, 255, 255] if is_good_crop else config.display.colors_by_camera[camera_index],
        "box_color": config.display.colors_by_camera[camera_index],
    }, one_or_more_cars_just_crossed


def run(config=None):
    config = config or AppConfig()
    embedder = embedding_utils.EmbeddingGenerator()
    cross_camera_matcher = CrossCameraMatcher(embedder, config.not_from_other_camera_masks_camera_1)
    visualizer = Visualizer(config)
    model = load_detection_model(config.model_path, confidence=0.02, iou=0.7, onnx_input_size=640)

    previous_centers_by_camera = [dict() for _ in config.video_paths]
    crossed_times_by_camera = [{}, {}]
    pending_click_by_camera = [None for _ in config.window_names]
    isolated_track_id_by_camera = [None for _ in config.window_names]
    frame_draw_data_by_camera = [None, None]

    _register_mouse_callbacks(config, pending_click_by_camera)

    captures = [cv2.VideoCapture(video_path) for video_path in config.video_paths]
    for cap in captures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame_index)

    fps = captures[0].get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    masks = _create_masks(captures, config.mask_points_by_camera)
    trackers = create_trackers_by_camera(config.model_path)

    paused = False
    step_next_frame = False
    current_frame_index = config.start_frame_index
    pause_at_frame_index = 1500
    paused_at_target_frame = False
    original_frames = [None, None]
    measured_fps = 0.0
    previous_frame_time = time.perf_counter()

    while True:
        processed_frame = False

        if not paused or step_next_frame:
            step_next_frame = False
            ret_and_frame_by_camera = [cap.read() for cap in captures]
            ret_by_camera = [ret for ret, _ in ret_and_frame_by_camera]
            if not all(ret_by_camera):
                break

            frame_by_camera = [frame for _, frame in ret_and_frame_by_camera]
            original_frames = [frame.copy() for frame in frame_by_camera]
            masked_frame_by_camera = [
                cv2.bitwise_and(frame, mask)
                for frame, mask in zip(frame_by_camera, masks)
            ]

            tracks_by_camera = predict_and_track(model, masked_frame_by_camera, trackers, original_frames, include_unconfirmed=False)
            current_frame_time = time.perf_counter()
            elapsed_seconds = current_frame_time - previous_frame_time
            previous_frame_time = current_frame_time
            if elapsed_seconds > 0:
                measured_fps = 1.0 / elapsed_seconds

            cross_camera_matcher.store_camera_1_embeddings(tracks_by_camera[1], original_frames[1])

            for camera_index in [0, 1]:
                draw_data = {"boxes": [],
                            "others": [],
                            "line": config.cross_lines[camera_index],
                            "frame_text": f"Frame {current_frame_index}",
                            "fps_text": f"FPS {measured_fps:.1f}"}

                one_or_more_cars_just_crossed = False

                for track in tracks_by_camera[camera_index]:
                    track_id = int(track[4])
                    if camera_index == 1 and not cross_camera_matcher.camera_1_track_is_relevant(track_id):
                        continue

                    box_draw_data, track_just_crossed = _process_track(camera_index, track,
                                                                       tracks_by_camera[camera_index], original_frames[camera_index],
                                                                       current_frame_index, delay_ms, config, cross_camera_matcher,
                                                                       previous_centers_by_camera, crossed_times_by_camera)
                    one_or_more_cars_just_crossed = one_or_more_cars_just_crossed or track_just_crossed
                    draw_data["boxes"].append(box_draw_data)

                if camera_index == 0 and one_or_more_cars_just_crossed:
                    cross_camera_matcher.refresh_camera_0_gallery()

                frame_draw_data_by_camera[camera_index] = draw_data

            processed_frame = True
            if current_frame_index == pause_at_frame_index and not paused_at_target_frame:
                paused = True
                paused_at_target_frame = True

        _handle_pending_clicks(pending_click_by_camera, isolated_track_id_by_camera, frame_draw_data_by_camera)

        key = cv2.waitKeyEx(1)
        key_code = key & 0xFF
        if key_code == ord("q"):
            break
        if key_code == ord(" "):
            paused = not paused
        elif paused and key in (83, 63235, 65363, 2555904):
            step_next_frame = True
        else:
            visualizer.handle_key(key_code)

        if processed_frame:
            current_frame_index += 1

        visualizer.draw(original_frames, frame_draw_data_by_camera, isolated_track_id_by_camera, cross_camera_matcher.best_matches_1)

    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
