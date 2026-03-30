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
EMBEDDING_SIMILARITY_THRESHOLD = 0.3
COLOR_SIMILARITY_THRESHOLD = 0.3
NUM_SHOW_POSSIBLE_OTHERS = 5
MODEL_PATH = r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt"


START_FRAME_INDEX = 1500
window_name_pair = ['c042', 'c041']

video_path_pair = [
    r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
    r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
]

CROSS_LINE_BOTH = [[(773, 175), (953, 256)],
                    [(227, 283), (731, 956)]]

MASK_PTS_PAIR = [[(4, 159), (228, 180), (489, 139), (696, 177), (1021, 119), (1279, 211), (1279, 2), (1, 4)],
                 [(181, 57), (438, 129), (527, 123), (749, 169), (1090, 144), (1251, 211), (1275, 2), (177, 3)]]

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
    is_important_1 = {}
    best_matches_1 = defaultdict(dict)
    pending_click_pair = [None for _ in window_name_pair]
    isolated_track_id_pair = [None for _ in window_name_pair]

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
                        if not track_id in is_important_1:
                            bottom_center = (int((x1 + x2) / 2), y2)
                            is_inside_bot_mask = (geometry_utils.point_inside_polygon(bottom_center, MASK_PTS_BOT_1) or
                                                  geometry_utils.point_inside_polygon(bottom_center, MASK_PTS_TOP_1))
                            is_important_1[track_id] = not is_inside_bot_mask



                    # append crops to their embedding histories
                    #
                    non_overlapping_crops_1 = []
                    non_overlapping_track_ids_1 = []
                    all_overlapping_1 = []
                    for t,track in enumerate(tracks):
                        track_id = int(track[4])
                        if is_important_1[track_id]:
                            x1, y1, x2, y2 = map(int, track[:4])
                            is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                            if not is_overlapping:
                                non_overlapping_track_ids_1.append(track_id)
                                non_overlapping_crops_1.append(orig_frame_pair[f][y1:y2, x1:x2])
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
                    if f == 1 and not is_important_1[track_id]:
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
                        updated_match = False
                        other_track_id = None
                        if not all_overlapping_1[t] and len(embedding_histories_1[track_id]):
                            query_embedding = np.mean(embedding_histories_1[track_id], axis=0)

                            if embedding_of_crossed_0.size == 0 or not embedding_of_crossed_0_map:
                                closest_embedding_idx, closest_embedding_score = None, None
                            else:
                                closest_embedding_idx, closest_embedding_score = embedding_utils.find_closest_embedding(query_embedding, embedding_of_crossed_0)

                            if closest_embedding_idx is not None and closest_embedding_score >= EMBEDDING_SIMILARITY_THRESHOLD:
                                query_color_hist = color_utils.compute_vehicle_color_histogram(orig_frame_pair[f][y1:y2, x1:x2])
                                other_track_id = embedding_of_crossed_0_map[closest_embedding_idx]
                                matched_color_idx, matched_color_score = color_utils.compare_histograms(
                                    query_color_hist,
                                    histograms_of_crossed_0.get(other_track_id, []))

                                if matched_color_score and matched_color_score >= COLOR_SIMILARITY_THRESHOLD:
                                    other_crop = None
                                    distributed_crops = general_utils.get_distributed_items(good_crops_per_ids_0[other_track_id]
                                                                                            if len(good_crops_per_ids_0[other_track_id])
                                                                                            else bad_crops_per_ids_0[other_track_id] )
                                    if matched_color_idx is not None and matched_color_idx < len(distributed_crops):
                                        other_crop = distributed_crops[matched_color_idx]
                                    elif distributed_crops:
                                        other_crop = distributed_crops[0]

                                    closest_total_score = closest_embedding_score * matched_color_score

                                    if track_id in crossed_times_pair[1]:
                                        elapsed_time = crossed_times_pair[1][track_id] - crossed_times_pair[0][other_track_id]
                                        # label = f"{label} t:{elapsed_time:.3f}"
                                    else:
                                        elapsed_time = -1.0

                                    do_record = False
                                    if track_id not in best_matches_1:
                                        do_record = True
                                    else:
                                        if other_track_id not in best_matches_1[track_id]:
                                            do_record = True
                                        else:
                                            if best_matches_1[track_id][other_track_id]['closest_total_score'] < closest_total_score:
                                                do_record = True
                                    if do_record:
                                        best_matches_1[track_id][other_track_id] = {'closest_total_score': closest_total_score,
                                                                                    'closest_embedding_score': closest_embedding_score,
                                                                                    'matched_color_score': matched_color_score,
                                                                                    'other_crop': other_crop,
                                                                                    'other_track_id': other_track_id,
                                                                                    'elapsed_time': elapsed_time}
                                        updated_match = True

                        # update the elapsed time, in case the car crossed and it wasn't calculated yet
                        #
                        if not updated_match:
                            if track_id in best_matches_1 and other_track_id in best_matches_1[track_id]:
                                if track_id in crossed_times_pair[1]:
                                    if best_matches_1[track_id][other_track_id]['elapsed_time'] == -1.0:
                                        best_matches_1[track_id][other_track_id]['elapsed_time'] = crossed_times_pair[1][track_id] - crossed_times_pair[0][other_track_id]

                        if track_id in best_matches_1:
                            matches = best_matches_1[track_id]
                            sorted_other_ids = sorted(list(matches.keys()), key=lambda x: matches[x]['closest_total_score'], reverse=True)
                            offset_y = 0
                            other_gap = 8
                            for other_id in sorted_other_ids[0:NUM_SHOW_POSSIBLE_OTHERS]:

                                match = best_matches_1[track_id][other_id]
                                elapsed_time = match['elapsed_time']
                                other_crop = match['other_crop']
                                other_label = (
                                    f"id:{match['other_track_id']}"
                                    f" score:{round(match['closest_embedding_score'], 4)}"
                                    f" color:{match['matched_color_score']:.2f}"
                                    f" t:{elapsed_time:.1f}"
                                )

                                crop_h, crop_w = other_crop.shape[:2]
                                box_w = max(1, x2 - x1)
                                target_w = max(1, int(round(box_w * 0.5)))
                                scale = target_w / max(1, crop_w)
                                target_h = max(1, int(round(crop_h * scale)))

                                frame_h, frame_w = orig_frame_pair[f].shape[:2]
                                paste_x2 = min(frame_w, x2)
                                base_y2 = min(frame_h, y2)
                                paste_y2 = min(frame_h, base_y2 + offset_y)
                                paste_x1 = max(0, paste_x2 - target_w)
                                paste_y1 = max(0, paste_y2 - target_h)

                                draw_data['others'].append({
                                    'track_id': track_id,
                                    'crop': other_crop,
                                    'target_w': target_w,
                                    'target_h': target_h,
                                    'paste_x1': paste_x1,
                                    'paste_y1': paste_y1,
                                    'paste_x2': paste_x2,
                                    'paste_y2': paste_y2,
                                    'other_track_id': match['other_track_id'],
                                    'label': other_label,
                                })

                                offset_y += target_h + other_gap
                    else:
                        raise Exception(f"unknown window name: {window_name_pair[f]}")
                    if f == 1:
                        label = f"{label} important: {is_important_1[track_id]}"
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

        if not paused:
            current_frame_index += 1



        # DRAW EVERYTHING
        #
        for f in [0, 1]:
            draw_frame[:] = orig_frame_pair[f][:]

            draw_data = frame_draw_data_pair[f]
            isolated_track_id = isolated_track_id_pair[f]
            cv2.line(draw_frame, draw_data['line'][0], draw_data['line'][1], (0, 0, 255), 2)

            for box in draw_data['boxes']:
                if isolated_track_id is not None and box['track_id'] != isolated_track_id:
                    continue
                x1, y1, x2, y2 = box['coords']
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), box['box_color'], 2)
                cv2.putText(draw_frame, box['label'], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box['label_color'], 2)

            for other in draw_data['others']:
                if isolated_track_id is not None and other['track_id'] != isolated_track_id:
                    continue
                resized_crop = cv2.resize(other['crop'], (other['target_w'], other['target_h']))
                if other['paste_y1'] < other['paste_y2'] and other['paste_x1'] < other['paste_x2']:
                    visible_crop = resized_crop[
                        other['target_h'] - (other['paste_y2'] - other['paste_y1']):,
                        other['target_w'] - (other['paste_x2'] - other['paste_x1']):,
                    ]
                    draw_frame[other['paste_y1']:other['paste_y2'], other['paste_x1']:other['paste_x2']] = visible_crop
                cv2.putText(draw_frame, other['label'], (other['paste_x1'], max(20, other['paste_y1'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_PAIR[0], 2)


            cv2.putText(draw_frame, draw_data['frame_text'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window_name_pair[f], draw_frame)


    for cap in cap_pair:
        cap.release()
    cv2.destroyAllWindows()
