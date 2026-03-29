import torch
from ultralytics import YOLO
import cv2
import numpy as np
import embedding_utils
import color_utils
import geometry_utils
from boxmot import OcSort
import time
from collections import defaultdict
import general_utils

import importlib
importlib.reload(geometry_utils)
importlib.reload(embedding_utils)
importlib.reload(color_utils)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COLORS_PAIR = [(255, 0, 0), (0, 255, 0)]
EMBEDDING_SIMILARITY_THRESHOLD = 0.3
COLOR_SIMILARITY_THRESHOLD = 0.3

START_FRAME_INDEX = 950
window_name_pair = ['c042', 'c041']

video_path_pair = [
    r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
    r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
]

CROSS_LINE_BOTH = [[(773, 175), (953, 256)],
                    [(227, 283), (731, 956)]]

MASK_PTS_PAIR = [[(0, 416), (721, 147), (963, 122), (1074, 197), (244, 959), (1, 955)],
                 [(4, 392), (336, 269), (766, 180), (1033, 160), (1144, 238), (556, 912), (334, 958), (5, 959)]]



def calculate_embedding_multiple(embedder, crops, distributed_count=16, return_mean=True):
    if distributed_count:
        distributed_crops = general_utils.get_distributed_items(crops, n=distributed_count)
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
    crops_per_ids_0 = defaultdict(list)
    embeddings_of_crossed_per_id_0 = {}
    histograms_of_crossed_0 = {}
    embedding_histories_1 = defaultdict(list)
    embedding_of_crossed_0_map = []
    embedding_of_crossed_0 = np.zeros(0)

    best_matches_1 = defaultdict(dict)

    for window_name in window_name_pair:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap_pair = [cv2.VideoCapture(video_path) for video_path in video_path_pair]
    for cap in cap_pair:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_INDEX)
    fps = cap_pair[0].get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    paused = False

    tracker_pair = [OcSort() for _ in video_path_pair]
    current_frame_index = START_FRAME_INDEX

    mask_pair = []
    for f, cap in enumerate(cap_pair):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        pts = np.array(MASK_PTS_PAIR[f], dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        mask_pair.append(mask)

    while True:
        loop_start = time.perf_counter()
        ret_and_frame_pair = [cap.read() for cap in cap_pair]
        ret_pair = [ret for ret, _ in ret_and_frame_pair]
        frame_pair = [frame for _, frame in ret_and_frame_pair]
        orig_frame_pair = [np.copy(frame) for _, frame in ret_and_frame_pair]
        masked_frame_pair = [cv2.bitwise_and(frame, mask) for frame, mask in zip(frame_pair, mask_pair)]

        if not all(ret_pair):
            break

        result_pair = model.predict(source=masked_frame_pair, verbose=False, conf=0.5)
        frame_draw_data_pair = []

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
                tracks = tracker_pair[f].update(detections, frame_pair[f])
            else:
                tracks = np.empty((0, 8), dtype=np.float32)

            one_or_more_cars_just_crossed = False

            if f == 1:
                # append crops of right camera to their embedding histories
                #
                non_overlapping_crops_1 = []
                non_overlapping_track_ids_1 = []
                all_overlapping_1 = []
                for t,track in enumerate(tracks):
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                    if not is_overlapping:
                        non_overlapping_track_ids_1.append(track_id)
                        non_overlapping_crops_1.append(orig_frame_pair[f][y1:y2, x1:x2])
                    all_overlapping_1.append(is_overlapping)

                combined_current_embeddings_1 = calculate_embedding_multiple(embedder, non_overlapping_crops_1, distributed_count=None, return_mean=False)
                if non_overlapping_crops_1:
                    for c, vector in enumerate(combined_current_embeddings_1):
                        track_id = non_overlapping_track_ids_1[c]
                        embedding_histories_1[track_id].append(vector)


            are_overlapping = []
            for t, track in enumerate(tracks):
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])

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
                            embeddings_of_crossed_per_id_0[track_id] = calculate_embedding_multiple(embedder, crops_per_ids_0[track_id])
                            histograms_of_crossed_0[track_id] = color_utils.calculate_histograms_multiple(crops_per_ids_0[track_id])


                prev_centers_pair[f][track_id] = (cx, cy)
                if track_id in crossed_times_pair[f]:
                    label = f"{label} crossed"

                recording_crop = False
                if f == 0:
                    # if we have good crops -> record them
                    #
                    is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                    width = x2 - x1
                    if not is_overlapping and width > 90:
                        crops_per_ids_0[track_id].append(geometry_utils.get_shrunk_crop(orig_frame_pair[f], x1, y1, x2, y2, scale=0.8))
                        recording_crop = True

                # c041: compare embeddings and histograms at each frame
                #
                elif f == 1:
                    updated_match = False
                    other_track_id = None
                    if not all_overlapping_1[t]:
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
                                distributed_crops = general_utils.get_distributed_items(crops_per_ids_0[other_track_id])
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
                        for other_id in sorted_other_ids[0:3]:

                            match = best_matches_1[track_id][other_id]
                            elapsed_time = match['elapsed_time']
                            other_crop = match['other_crop']

                            label = (
                                f"{label} score: {round(match['closest_embedding_score'], 4)}"
                                f" color: {match['matched_color_score']:.2f}"
                            )
                            label = f"{label} t:{elapsed_time:.1f}"

                            crop_h, crop_w = other_crop.shape[:2]
                            box_w = max(1, x2 - x1)
                            target_w = max(1, int(round(box_w * 0.5)))
                            scale = target_w / max(1, crop_w)
                            target_h = max(1, int(round(crop_h * scale)))

                            frame_h, frame_w = frame_pair[f].shape[:2]
                            paste_x2 = min(frame_w, x2)
                            base_y2 = min(frame_h, y2)
                            paste_y2 = min(frame_h, base_y2 + offset_y)
                            paste_x1 = max(0, paste_x2 - target_w)
                            paste_y1 = max(0, paste_y2 - target_h)

                            draw_data['others'].append({
                                'crop': other_crop,
                                'target_w': target_w,
                                'target_h': target_h,
                                'paste_x1': paste_x1,
                                'paste_y1': paste_y1,
                                'paste_x2': paste_x2,
                                'paste_y2': paste_y2,
                                'other_track_id': match['other_track_id'],
                            })

                            offset_y += target_h + other_gap
                else:
                    raise Exception(f"unknown window name: {window_name_pair[f]}")
                draw_data['boxes'].append({
                    'coords': (x1, y1, x2, y2),
                    'label': label,
                    'label_color': [255,255,255] if recording_crop else COLORS_PAIR[f],
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

            frame_draw_data_pair.append(draw_data)


        # DRAW EVERYTHING
        #
        for f in [0, 1]:
            draw_data = frame_draw_data_pair[f]
            cv2.line(frame_pair[f], draw_data['line'][0], draw_data['line'][1], (0, 0, 255), 2)

            for other in draw_data['others']:
                resized_crop = cv2.resize(other['crop'], (other['target_w'], other['target_h']))
                if other['paste_y1'] < other['paste_y2'] and other['paste_x1'] < other['paste_x2']:
                    visible_crop = resized_crop[
                        other['target_h'] - (other['paste_y2'] - other['paste_y1']):,
                        other['target_w'] - (other['paste_x2'] - other['paste_x1']):,
                    ]
                    frame_pair[f][other['paste_y1']:other['paste_y2'], other['paste_x1']:other['paste_x2']] = visible_crop
                cv2.putText(frame_pair[f], f"id:{other['other_track_id']}", (other['paste_x1'], max(20, other['paste_y1'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_PAIR[0], 2)

            for box in draw_data['boxes']:
                x1, y1, x2, y2 = box['coords']
                cv2.rectangle(frame_pair[f], (x1, y1), (x2, y2), box['box_color'], 2)
                cv2.putText(frame_pair[f], box['label'], (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box['label_color'], 2)

            cv2.putText(frame_pair[f], draw_data['frame_text'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window_name_pair[f], frame_pair[f])

        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            elapsed_ms = int(round((time.perf_counter() - loop_start) * 1000.0))
            wait_ms = max(1, delay_ms - elapsed_ms)
            key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

        if not paused:
            current_frame_index += 1

    for cap in cap_pair:
        cap.release()
    cv2.destroyAllWindows()
