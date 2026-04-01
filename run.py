import torch
from ultralytics import YOLO
import cv2
import numpy as np
import embedding_utils
import color_utils
import geometry_utils
from boxmot import BotSort
import time
from collections import defaultdict
import general_utils
from pathlib import Path

import importlib
importlib.reload(geometry_utils)
importlib.reload(embedding_utils)
importlib.reload(color_utils)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORS_PAIR = [(255, 0, 0), (0, 255, 0)]

INFERENCE_IGNORE_AREA_COLOR = (255, 0, 0)
INFERENCE_IGNORE_AREA_ALPHA = 0.5

NOT_FROM_OTHER_CAMERA_AREA_COLOR = (0, 0, 255)
NOT_FROM_OTHER_CAMERA_AREA_ALPHA = 0.5

EMBEDDING_SIMILARITY_THRESHOLD = 0.0
COLOR_SIMILARITY_THRESHOLD = 0.0
MODEL_PATH = r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt"


START_FRAME_INDEX = 0
window_name_pair = ['c042', 'c041']

video_path_pair = [
    r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
    r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
]

CROSS_LINE_BOTH = [[(773, 175), (953, 256)],
                    [(227, 283), (731, 956)]]

MASK_PTS_PAIR = [[(1278, 493), (961, 256), (1101, 163), (1027, 101), (881, 126), (684, 165), (499, 128), (304, 142), (168, 145), (7, 222), (57, 290), (2, 352), (0, 7), (1278, 4)],
                 [(1, 293), (146, 205), (105, 124), (247, 86), (424, 135), (534, 116), (728, 168), (1011, 148), (1133, 145), (1199, 197), (1087, 273), (1138, 361), (1278, 412), (1276, 3), (5, 3)]]

MASK_PTS_BOT_1 = [(657, 948), (1083, 286), (1278, 419), (1277, 956)]
MASK_PTS_TOP_1 = [(2, 370), (536, 188), (888, 162), (1277, 199), (1275, 5), (2, 4)]

came_from_other_directions_1 = {}

def calculate_embedding_multiple(embedder, crops, distributed_count=16, return_mean=True):
    if distributed_count:
        distributed_crops = general_utils.get_distributed_items(crops, n=distributed_count) # suddenly crops are empty
    else:
        distributed_crops = crops

    vector = embedder.get_embeddings(distributed_crops)
    if return_mean:
        mean_vector = np.mean(vector, axis=0)
        return mean_vector
    else:
        return vector




def run():
    embedder = embedding_utils.EmbeddingGenerator()
    embedding_size = embedder.embedding_dim
    model = YOLO(r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt") # started with "yolo11m.pt"

    prev_centers_pair = [dict() for _ in video_path_pair]
    crossed_times_pair = [{}, {}]
    good_crops_per_ids_0 = defaultdict(list)
    bad_crops_per_ids_0 = defaultdict(list)
    embeddings_of_crossed_per_id_0 = {}
    histograms_of_crossed_0 = {}
    embedding_histories_1 = defaultdict(list)
    embedding_of_crossed_0_map = []
    embedding_of_crossed_0 = np.zeros(0)
    comes_from_other_camera_1 = {}
    best_matches_1 = defaultdict(list)
    pending_click_pair = [None for _ in window_name_pair]
    isolated_track_id_pair = [None for _ in window_name_pair]
    num_other_matches_to_show = 5
    show_inference_ignore_area = False
    show_not_from_other_camera_area = False

    for f, window_name in enumerate(window_name_pair):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        def mouse_callback(event, x, y, flags, param, frame_idx=f):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_click_pair[frame_idx] = (x, y)

        cv2.setMouseCallback(window_name, mouse_callback)

    cap_pair = [cv2.VideoCapture(video_path) for video_path in video_path_pair]
    for cap in cap_pair:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_INDEX)
    fps = cap_pair[0].get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    paused = False

    tracker_pair = [BotSort(reid_weights=Path(MODEL_PATH),
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
                            frame_rate=30)
                     for _ in video_path_pair]

    current_frame_index = START_FRAME_INDEX

    mask_pair = []
    for f, cap in enumerate(cap_pair):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        pts = np.array(MASK_PTS_PAIR[f], dtype=np.int32)
        cv2.fillPoly(mask, [pts], (0, 0, 0))
        mask_pair.append(mask)

    frame_draw_data_pair = [None, None]
    draw_frame = np.empty((frame_height, frame_width, 3), dtype=np.uint8)
    while True:
        loop_start = time.perf_counter()

        if not paused:
            ret_and_frame_pair = [cap.read() for cap in cap_pair]
            ret_pair = [ret for ret, _ in ret_and_frame_pair]
            frame_pair = [frame for _, frame in ret_and_frame_pair]
            orig_frame_pair = [np.copy(frame) for _, frame in ret_and_frame_pair]
            masked_frame_pair = [cv2.bitwise_and(frame, mask) for frame, mask in zip(frame_pair, mask_pair)]

            if not all(ret_pair):
                break

            result_pair = model.predict(source=masked_frame_pair, verbose=False, conf=0.02)

            for f in [0,1]:
                draw_data = {
                    'boxes': [],
                    'others': [],
                    'line': CROSS_LINE_BOTH[f],
                    'frame_text': f"Frame {current_frame_index}",
                }

                if result_pair[f].boxes is not None and len(result_pair[f].boxes) > 0:
                    boxes = result_pair[f].boxes.xyxy.cpu().numpy()
                    confs = result_pair[f].boxes.conf.cpu().numpy().reshape(-1, 1)
                    clss = result_pair[f].boxes.cls.cpu().numpy().reshape(-1, 1)

                    detections = np.hstack((boxes, confs, clss)).astype(np.float32)
                    tracks = tracker_pair[f].update(detections, orig_frame_pair[f])
                else:
                    tracks = np.empty((0, 8), dtype=np.float32)

                one_or_more_cars_just_crossed = False

                if f == 1:
                    for t,track in enumerate(tracks):
                        x1, y1, x2, y2 = map(int, track[:4])
                        track_id = int(track[4])
                        if not track_id in comes_from_other_camera_1:
                            bottom_center = (int((x1 + x2) / 2), y2)
                            is_inside_bot_mask = (geometry_utils.point_inside_polygon(bottom_center, MASK_PTS_BOT_1) or
                                                  geometry_utils.point_inside_polygon(bottom_center, MASK_PTS_TOP_1))
                            comes_from_other_camera_1[track_id] = not is_inside_bot_mask



                    # append crops to their embedding histories
                    #
                    non_overlapping_crops_1 = []
                    non_overlapping_track_ids_1 = []
                    all_overlapping_1 = []
                    for t,track in enumerate(tracks):
                        track_id = int(track[4])
                        if comes_from_other_camera_1[track_id]:
                            x1, y1, x2, y2 = map(int, track[:4])
                            is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                            if not is_overlapping:
                                non_overlapping_track_ids_1.append(track_id)
                                non_overlapping_crops_1.append(geometry_utils.get_shrunk_crop(orig_frame_pair[f], x1, y1, x2, y2, scale=0.8))
                            all_overlapping_1.append(is_overlapping)
                        else:
                            all_overlapping_1.append(None)

                    combined_current_embeddings_1 = calculate_embedding_multiple(embedder, non_overlapping_crops_1, distributed_count=None, return_mean=False)
                    if non_overlapping_crops_1:
                        for c, vector in enumerate(combined_current_embeddings_1):
                            track_id = non_overlapping_track_ids_1[c]
                            embedding_histories_1[track_id].append(vector)


                are_overlapping = []
                for t, track in enumerate(tracks):
                    track_id = int(track[4])
                    if f == 1 and not comes_from_other_camera_1[track_id]:
                        continue

                    x1, y1, x2, y2 = map(int, track[:4])

                    label = f"ID {track_id}"

                    # checking if crossed
                    #
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    prev = prev_centers_pair[f].get(track_id)
                    if prev and track_id not in crossed_times_pair[f]:
                        if geometry_utils.segments_intersect(prev, (cx, cy), CROSS_LINE_BOTH[f][0], CROSS_LINE_BOTH[f][1]):
                            one_or_more_cars_just_crossed = True
                            crossed_times_pair[f][track_id] = current_frame_index * (delay_ms / 1000.0)

                            # in the left camera, we calculate their embeddings and histograms whenever they crossed the line
                            #
                            if f == 0:
                                crops = good_crops_per_ids_0[track_id] if len(good_crops_per_ids_0[track_id]) else bad_crops_per_ids_0[track_id]
                                embeddings_of_crossed_per_id_0[track_id] = calculate_embedding_multiple(embedder, crops)
                                histograms_of_crossed_0[track_id] = color_utils.calculate_histograms_multiple(crops)


                    prev_centers_pair[f][track_id] = (cx, cy)
                    if track_id in crossed_times_pair[f]:
                        label = f"{label} crossed"

                    is_good_crop = False
                    if f == 0:
                        # if we have good crops -> record them
                        #
                        is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                        width = x2 - x1
                        if not is_overlapping and width > 90:
                            good_crops_per_ids_0[track_id].append(geometry_utils.get_shrunk_crop(orig_frame_pair[f], x1, y1, x2, y2, scale=0.8))
                            is_good_crop = True
                        else:
                            bad_crops_per_ids_0[track_id].append(geometry_utils.get_shrunk_crop(orig_frame_pair[f], x1, y1, x2, y2, scale=0.8))

                    # c041: compare embeddings and histograms at each frame
                    #
                    elif f == 1:
                        if not all_overlapping_1[t] and len(embedding_histories_1[track_id]):
                            query_embedding = np.mean(embedding_histories_1[track_id], axis=0)

                            if embedding_of_crossed_0.size == 0 or not embedding_of_crossed_0_map:
                                closest_embedding_indices, embedding_scores = [], []
                            else:
                                closest_embedding_indices, embedding_scores = embedding_utils.find_closest_embeddings(query_embedding, embedding_of_crossed_0)


                            for closest_embedding_index, embedding_score in zip(closest_embedding_indices, embedding_scores):
                                other_track_id = embedding_of_crossed_0_map[closest_embedding_index]
                                if len(good_crops_per_ids_0[other_track_id]):
                                    good_crops_per_ids_0[other_track_id].sort(key=lambda xx: xx.shape[1])
                                    other_draw_crop = good_crops_per_ids_0[other_track_id][-1]
                                else:
                                    bad_crops_per_ids_0[other_track_id].sort(key=lambda xx: xx.shape[1])
                                    other_draw_crop = bad_crops_per_ids_0[other_track_id][-1]

                                if track_id in crossed_times_pair[1]:
                                    elapsed_time = crossed_times_pair[1][track_id] - crossed_times_pair[0][other_track_id]
                                else:
                                    elapsed_time = -1.0

                                do_append = True
                                for m, _match_data in enumerate(best_matches_1[track_id]):
                                    if _match_data['other_track_id'] == other_track_id:
                                        _match_data['embedding_score'] = embedding_score
                                        do_append = False
                                        break
                                if do_append:
                                    best_matches_1[track_id].append({'embedding_score': embedding_score,
                                                                    'other_draw_crop': other_draw_crop,
                                                                    'other_track_id': other_track_id,
                                                                    'elapsed_time': -1.0})


                        # update the elapsed time, in case the car crossed and it wasn't calculated yet
                        #
                        if track_id in best_matches_1:
                            if track_id in crossed_times_pair[1]:
                                for _match_data in best_matches_1[track_id]:
                                        if _match_data['elapsed_time'] == -1.0:
                                            other_track_id = _match_data['other_track_id']
                                            _match_data['elapsed_time'] = crossed_times_pair[1][track_id] - crossed_times_pair[0][other_track_id]


                            best_matches_1[track_id].sort(key=lambda x: x['embedding_score'], reverse=True)
                    else:
                        raise Exception(f"unknown window name: {window_name_pair[f]}")

                    draw_data['boxes'].append({
                        'track_id': track_id,
                        'coords': (x1, y1, x2, y2),
                        'label': label,
                        'label_color': [255,255,255] if is_good_crop else COLORS_PAIR[f],
                        'box_color': COLORS_PAIR[f],
                    })


                if f == 0 and one_or_more_cars_just_crossed:
                    # get galleries of left camera:
                    #
                    embedding_of_crossed_0 = np.zeros((len(embeddings_of_crossed_per_id_0), embedding_size), dtype='float64')
                    embedding_of_crossed_0_map.clear()
                    for t, other_track_id in enumerate(sorted(embeddings_of_crossed_per_id_0.keys())):
                        embedding_of_crossed_0[t] = embeddings_of_crossed_per_id_0[other_track_id]
                        embedding_of_crossed_0_map.append(other_track_id)

                frame_draw_data_pair[f] = draw_data


        for f in [0, 1]:
            pending_click = pending_click_pair[f]
            if pending_click is None:
                continue

            clicked_box = None
            for box in reversed(frame_draw_data_pair[f]['boxes']):
                if geometry_utils.point_inside_box(pending_click, box['coords']):
                    clicked_box = box
                    break

            if clicked_box is not None:
                isolated_track_id_pair[f] = clicked_box['track_id']
            elif isolated_track_id_pair[f] is not None:
                isolated_track_id_pair[f] = None

            pending_click_pair[f] = None


        elapsed_ms = int(round((time.perf_counter() - loop_start) * 1000.0))
        wait_ms = max(1, delay_ms - elapsed_ms)
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        elif ord("0") <= key <= ord("9"):
            num_other_matches_to_show = key - ord("0")
        elif key in (ord("m"), ord("M")):
            show_inference_ignore_area = not show_inference_ignore_area
        elif key in (ord("o"), ord("O")):
            show_not_from_other_camera_area = not show_not_from_other_camera_area

        if not paused:
            current_frame_index += 1



        # DRAW EVERYTHING
        #
        for f in [0, 1]:
            draw_frame[:] = orig_frame_pair[f][:]

            draw_data = frame_draw_data_pair[f]
            isolated_track_id = isolated_track_id_pair[f]
            overlay = draw_frame.copy()

            if show_not_from_other_camera_area:
                cv2.fillPoly(overlay, [np.array(MASK_PTS_PAIR[f], dtype=np.int32)], NOT_FROM_OTHER_CAMERA_AREA_COLOR)
                cv2.addWeighted(overlay, NOT_FROM_OTHER_CAMERA_AREA_ALPHA, draw_frame, 1 - NOT_FROM_OTHER_CAMERA_AREA_ALPHA, 0, draw_frame)

            if f == 1 and show_inference_ignore_area:
                overlay = draw_frame.copy()
                cv2.fillPoly(overlay, [np.array(MASK_PTS_BOT_1, dtype=np.int32)], INFERENCE_IGNORE_AREA_COLOR)
                cv2.fillPoly(overlay, [np.array(MASK_PTS_TOP_1, dtype=np.int32)], INFERENCE_IGNORE_AREA_COLOR)
                cv2.addWeighted(overlay, INFERENCE_IGNORE_AREA_ALPHA, draw_frame, 1 - INFERENCE_IGNORE_AREA_ALPHA, 0, draw_frame)

            cv2.line(draw_frame, draw_data['line'][0], draw_data['line'][1], (0, 0, 255), 2)

            for box in draw_data['boxes']:
                if isolated_track_id is not None and box['track_id'] != isolated_track_id:
                    continue
                x1, y1, x2, y2 = box['coords']
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), box['box_color'], 2)
                cv2.putText(draw_frame, box['label'], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box['label_color'], 2)

                if f != 1 or box['track_id'] not in best_matches_1:
                    continue

                best_matches_1[box['track_id']].sort(key=lambda x: x['embedding_score'], reverse=True)
                panel_items = []
                panel_width = 0
                panel_height = 0
                top_matches = best_matches_1[box['track_id']][0:num_other_matches_to_show]
                for _match_data in reversed(top_matches):
                    other_track_id = _match_data['other_track_id']
                    elapsed_time = _match_data['elapsed_time']
                    other_draw_crop = _match_data['other_draw_crop']
                    other_label = (
                        f"id:{other_track_id}"
                        f" score:{_match_data['embedding_score']:.3f}"
                        f" t:{elapsed_time:.1f}"
                    )

                    crop_h, crop_w = other_draw_crop.shape[:2]
                    box_w = max(1, x2 - x1)
                    target_w = max(1, int(round(box_w * 0.5)))
                    scale = target_w / max(1, crop_w)
                    target_h = max(1, int(round(crop_h * scale)))

                    panel_items.append({
                        'crop': other_draw_crop,
                        'target_w': target_w,
                        'target_h': target_h,
                        'label': other_label,
                    })
                    panel_width = max(panel_width, target_w)
                    panel_height += target_h

                if panel_width == 0 or panel_height == 0:
                    continue

                panel = np.zeros((panel_height, panel_width, 3), dtype=draw_frame.dtype)
                text_items = []
                panel_y = 0
                for item in panel_items:
                    resized_crop = cv2.resize(item['crop'], (item['target_w'], item['target_h']))
                    panel_x1 = panel_width - item['target_w']
                    panel_x2 = panel_width
                    panel_y1 = panel_y
                    panel_y2 = panel_y + item['target_h']
                    panel[panel_y1:panel_y2, panel_x1:panel_x2] = resized_crop
                    text_items.append({
                        'label': item['label'],
                        'x': panel_x1,
                        'y': panel_y2 - 3,
                    })
                    panel_y = panel_y2

                frame_h, frame_w = draw_frame.shape[:2]
                paste_x2 = min(frame_w, x2)
                paste_y2 = min(frame_h, y2)
                paste_x1 = max(0, paste_x2 - panel_width)
                paste_y1 = max(0, paste_y2 - panel_height)
                visible_w = paste_x2 - paste_x1
                visible_h = paste_y2 - paste_y1

                if visible_w <= 0 or visible_h <= 0:
                    continue

                visible_panel = panel[panel_height - visible_h:, panel_width - visible_w:]
                draw_frame[paste_y1:paste_y2, paste_x1:paste_x2] = visible_panel

                hidden_x = panel_width - visible_w
                hidden_y = panel_height - visible_h
                for text_item in text_items:
                    text_x = text_item['x'] - hidden_x
                    text_y = text_item['y'] - hidden_y
                    if text_y < 0 or text_y >= visible_h:
                        continue
                    cv2.putText(
                        draw_frame,
                        text_item['label'],
                        (paste_x1 + max(0, text_x), paste_y1 + text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        COLORS_PAIR[0],
                        2,
                    )

            legend_lines = [
                draw_data['frame_text'],
                (
                    f"0-9 matches:{num_other_matches_to_show}  "
                    f"M inference-ignore:{'on' if show_inference_ignore_area else 'off'}  "
                    f"O not-from-other-camera:{'on' if show_not_from_other_camera_area else 'off'}"
                ),
            ]
            for line_idx, legend_line in enumerate(legend_lines):
                y = 30 + line_idx * 28
                cv2.putText(draw_frame, legend_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window_name_pair[f], draw_frame)


    for cap in cap_pair:
        cap.release()
    cv2.destroyAllWindows()
