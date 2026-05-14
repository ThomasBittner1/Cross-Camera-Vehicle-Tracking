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


def _point_side_of_line(point, line):
    x, y = point
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def _track_center(track):
    x1, y1, x2, y2 = map(int, track[:4])
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _crossed_line(previous_center, current_center, crossing_line, directional=False):
    if previous_center is None:
        return False

    crossed = geometry_utils.segments_intersect(
        previous_center,
        current_center,
        crossing_line[0],
        crossing_line[1],
    )
    if not crossed or not directional:
        return crossed

    return (
        _point_side_of_line(previous_center, crossing_line) < 0
        and _point_side_of_line(current_center, crossing_line) > 0
    )


def run(config=None):
    config = config or AppConfig()
    embedder = embedding_utils.EmbeddingGenerator()
    cross_camera_matcher = CrossCameraMatcher(embedder, config.not_from_other_camera_masks_query_camera)
    visualizer = Visualizer(config)
    model = load_detection_model(config.model_path, confidence=0.02, iou=0.7, onnx_input_size=640)

    previous_centers_by_camera = [dict() for _ in config.video_paths]
    exited_track_ids_source = set()
    source_track_last_seen_frame = {}
    registered_source_track_ids = set()
    exited_times_source = {}
    crossed_times_query = {}
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
    pause_at_frame_index = 1450
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

            cross_camera_matcher.store_query_camera_embeddings(tracks_by_camera[1], original_frames[1])
            current_source_track_ids = {int(track[4]) for track in tracks_by_camera[0]}

            source_draw_data = {"boxes": [],
                                "others": [],
                                "line": None,
                                "exit_lines": config.disappear_lines_source,
                                "frame_text": f"Frame {current_frame_index}",
                                "fps_text": f"FPS {measured_fps:.1f}"}
            for track in tracks_by_camera[0]:
                track_id = int(track[4])
                x1, y1, x2, y2 = map(int, track[:4])
                previous_center = previous_centers_by_camera[0].get(track_id)
                current_center = _track_center(track)
                is_good_crop = False

                source_track_last_seen_frame[track_id] = current_frame_index
                for exit_line in config.disappear_lines_source:
                    if track_id in exited_track_ids_source:
                        continue

                    if _crossed_line(previous_center, current_center, exit_line, directional=True):
                        exited_track_ids_source.add(track_id)

                previous_centers_by_camera[0][track_id] = current_center

                min_side_length = min(abs(x2 - x1), abs(y2 - y1))
                if min_side_length > 40:
                    is_good_crop = cross_camera_matcher.record_source_camera_crop(
                        track,
                        tracks_by_camera[0],
                        original_frames[0])

                source_draw_data["boxes"].append({"track_id": track_id,
                                                  "coords": (x1, y1, x2, y2),
                                                  "label": f"{track_id}",
                                                  "label_color": [255, 255, 255] if is_good_crop else config.display.colors_by_camera[0],
                                                  "box_color": config.display.colors_by_camera[0]})

            frame_draw_data_by_camera[0] = source_draw_data

            query_draw_data = {"boxes": [],
                               "others": [],
                               "line": config.entry_line_query,
                               "exit_lines": [],
                               "frame_text": f"Frame {current_frame_index}",
                               "fps_text": f"FPS {measured_fps:.1f}"}
            for track in tracks_by_camera[1]:
                track_id = int(track[4])
                if not cross_camera_matcher.query_camera_track_is_relevant(track_id):
                    continue

                x1, y1, x2, y2 = map(int, track[:4])
                previous_center = previous_centers_by_camera[1].get(track_id)
                current_center = _track_center(track)
                label = f"{track_id}"

                if _crossed_line(previous_center, current_center, config.entry_line_query):
                    if track_id not in crossed_times_query:
                        crossed_times_query[track_id] = current_frame_index * (delay_ms / 1000.0)

                previous_centers_by_camera[1][track_id] = current_center

                if track_id in crossed_times_query:
                    label = f"{label} crossed"

                cross_camera_matcher.update_query_camera_matches(track_id, exited_times_source, crossed_times_query)
                cross_camera_matcher.update_query_camera_elapsed_times(track_id, exited_times_source, crossed_times_query)

                query_draw_data["boxes"].append({"track_id": track_id,
                                                 "coords": (x1, y1, x2, y2),
                                                 "label": label,
                                                 "label_color": config.display.colors_by_camera[1],
                                                 "box_color": config.display.colors_by_camera[1]})
            frame_draw_data_by_camera[1] = query_draw_data

            source_gallery_changed = False
            for track_id, last_seen_frame in source_track_last_seen_frame.items():
                if track_id in current_source_track_ids or track_id in registered_source_track_ids:
                    continue
                registered_source_track_ids.add(track_id)
                if track_id in exited_track_ids_source:
                    continue

                exited_times_source[track_id] = last_seen_frame * (delay_ms / 1000.0)
                cross_camera_matcher.record_embeddings(track_id)
                source_gallery_changed = True

            if source_gallery_changed:
                cross_camera_matcher.refresh_source_camera_gallery()

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

        visualizer.draw(original_frames, frame_draw_data_by_camera, isolated_track_id_by_camera, cross_camera_matcher.get_best_matches())

    for cap in captures:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
