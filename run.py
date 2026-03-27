import torch
from ultralytics import YOLO
import cv2
import numpy as np
import embedding_utils
import geometry_utils
from boxmot import OcSort
import time
from collections import defaultdict

embedder = embedding_utils.EmbeddingGenerator()


import importlib
importlib.reload(geometry_utils)
importlib.reload(embedding_utils)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c042_cross_line = [(773, 175), (953, 256)]
c041_cross_line = [(260, 331), (802, 906)]
# masks are created with draw_mask.py

mask_c042 = [(0, 416), (721, 147), (963, 122), (1074, 197), (244, 959), (1, 955)]
mask_c041 = [(4, 392), (336, 269), (766, 180), (1033, 160), (1144, 238), (556, 912), (334, 958), (5, 959)]

c041_other_best_crops = {}
c042_other_best_embedding_distance = {}
c042_other_best_color_score = {}

EMBEDDING_SIZE = 2048
EMBEDDING_SIMILARITY_THRESHOLD = 0.3
COLOR_SIMILARITY_THRESHOLD = 0.3


def calculate_embedding_exited_car(crops, distributed_count=16, return_mean=True):
    if distributed_count:
        distributed_crops = geometry_utils.get_distributed_items(crops, n=distributed_count)
    else:
        distributed_crops = crops

    vector = embedder.get_embeddings(distributed_crops)
    if return_mean:
        mean_vector = np.mean(vector, axis=0)
        return mean_vector
    else:
        return vector


def calculate_embedding_single(crop):
    vector = embedder.get_embeddings([crop])[0]
    return vector


def calculate_color_histograms_exited_car(crops):
    distributed_crops = geometry_utils.get_distributed_items(crops)
    histograms = []
    for crop in distributed_crops:
        histogram = embedding_utils.compute_vehicle_color_histogram(crop)
        if histogram is not None:
            histograms.append(histogram)
    return histograms


def calculate_color_histogram_single(crop):
    return embedding_utils.compute_vehicle_color_histogram(crop)



def run():
    model = YOLO(r"C:\ComputerVision\car_multicamera\runs\train10\weights\best.pt")
    # model = YOLO("yolo11m.pt")

    video_paths = [
        r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
        r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
    ]
    window_names = ['c042', 'c041']
    for window_name in window_names:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 10.0
    delay_ms = max(1, int(round(1000.0 / fps)))
    paused = False

    trackers = [OcSort() for _ in video_paths]
    frame_index = 0
    prev_centers = [dict() for _ in video_paths]
    crossed_ids = [set() for _ in video_paths]
    crops_per_ids = [defaultdict(list) for _ in video_paths]
    embedding_vectors_of_crossed_c042 = {}
    color_histograms_of_crossed_c042 = {}
    embedding_vectors_of_crossed_c041 = {}

    embedding_histories_1 = defaultdict(list)

    colors = [(255, 0, 0), (0, 255, 0)]

    masks = []
    for cap, pts in zip(caps, [mask_c042, mask_c041]):
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        masks.append(mask)

    while True:
        loop_start = time.perf_counter()
        rets_and_frames = [cap.read() for cap in caps]
        rets = [ret for ret, _ in rets_and_frames]
        frames = [frame for _, frame in rets_and_frames]
        orig_frames = [np.copy(frame) for _, frame in rets_and_frames]
        masked_frames = [cv2.bitwise_and(frame, mask) for frame, mask in zip(frames, masks)]
        # frames = masked_frames
        if not all(rets):
            break

        results = model.predict(
            source=masked_frames,
            verbose=False,
            conf=0.5
        )


        for f, (frame, result, tracker) in enumerate(zip(frames, results, trackers)):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
                clss = result.boxes.cls.cpu().numpy().reshape(-1, 1)

                detections = np.hstack((boxes, confs, clss)).astype(np.float32)
                tracks = tracker.update(detections, frame)
            else:
                tracks = np.empty((0, 8), dtype=np.float32)


            # right camera: get galleries of left camera, and combined crop of right camera
            if window_names[f] == 'c041':
                gallery_c042 = np.zeros((len(embedding_vectors_of_crossed_c042), EMBEDDING_SIZE), dtype='float64')
                gallery_c042_map = []
                for t, other_track_id in enumerate(sorted(embedding_vectors_of_crossed_c042.keys())):
                    gallery_c042[t] = embedding_vectors_of_crossed_c042[other_track_id]
                    gallery_c042_map.append(other_track_id)

                # append to embedding history
                #
                non_overlapping_crops_1 = []
                non_overlapping_track_ids = []
                all_overlapping_1 = []
                for t,track in enumerate(tracks):
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[4])
                    is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                    if not is_overlapping:
                        non_overlapping_track_ids.append(track_id)
                        non_overlapping_crops_1.append(orig_frames[f][y1:y2, x1:x2])

                    all_overlapping_1.append(is_overlapping)

                all_current_embeddings_1 = calculate_embedding_exited_car(non_overlapping_crops_1, distributed_count=None, return_mean=False)
                if non_overlapping_crops_1:
                    for tt, vector in enumerate(all_current_embeddings_1):
                        track_id = non_overlapping_track_ids[tt]
                        embedding_histories_1[track_id].append(vector)


            are_overlapping = []
            for t, track in enumerate(tracks):
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])

                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[f], 2)
                label = f"ID {track_id}"

                if f == 0:
                    is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
                    if not is_overlapping:
                        crops_per_ids[f][track_id].append(orig_frames[f][y1:y2, x1:x2])

                # c042: if car crosses red line -> record embeddings and histograms
                #
                if f == 0:
                    cv2.line(frame, c042_cross_line[0], c042_cross_line[1], (0, 0, 255), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    prev = prev_centers[f].get(track_id)
                    if prev and track_id not in crossed_ids[f]:
                        if geometry_utils.segments_intersect(prev, (cx, cy), c042_cross_line[0], c042_cross_line[1]):
                            crossed_ids[f].add(track_id)
                            embedding_vectors_of_crossed_c042[track_id] = calculate_embedding_exited_car(crops_per_ids[f][track_id])
                            color_histograms_of_crossed_c042[track_id] = calculate_color_histograms_exited_car(crops_per_ids[f][track_id])

                    prev_centers[f][track_id] = (cx, cy)
                    if track_id in crossed_ids[f]:
                        label = f"{label} crossed"


                # c041: always compare embeddings and histograms
                #
                elif f == 1:
                    if not all_overlapping_1[t]:
                        query_embedding = np.mean(embedding_histories_1[track_id], axis=0)

                        if gallery_c042.size == 0 or not gallery_c042_map:
                            closest_embedding_idx, closest_embedding_score = None, None
                        else:
                            closest_embedding_idx, closest_embedding_score = embedding_utils.find_closest_embedding(query_embedding, gallery_c042)

                        if closest_embedding_idx is not None and closest_embedding_score >= EMBEDDING_SIMILARITY_THRESHOLD:
                            query_color_hist = calculate_color_histogram_single(orig_frames[f][y1:y2, x1:x2])
                            other_track_id = gallery_c042_map[closest_embedding_idx]
                            matched_color_idx, matched_color_score = embedding_utils.compare_color_histograms(
                                query_color_hist,
                                color_histograms_of_crossed_c042.get(other_track_id, []),
                            )

                            if matched_color_score and matched_color_score >= COLOR_SIMILARITY_THRESHOLD:
                                distributed_crops = geometry_utils.get_distributed_items(crops_per_ids[0][other_track_id])
                                if matched_color_idx is not None and matched_color_idx < len(distributed_crops):
                                    c041_other_best_crops[track_id] = distributed_crops[matched_color_idx]
                                elif distributed_crops:
                                    c041_other_best_crops[track_id] = distributed_crops[0]
                                c042_other_best_embedding_distance[track_id] = closest_embedding_score
                                c042_other_best_color_score[track_id] = matched_color_score

                            if track_id in c041_other_best_crops:
                                label = (
                                    f"{label} score: {round(c042_other_best_embedding_distance[track_id], 4)}"
                                    f" color: {round(c042_other_best_color_score[track_id], 4)}"
                                )
                                other_crop = c041_other_best_crops[track_id]
                                crop_h, crop_w = other_crop.shape[:2]
                                box_w = max(1, x2 - x1)
                                target_w = max(1, int(round(box_w * 0.5)))
                                scale = target_w / max(1, crop_w)
                                target_h = max(1, int(round(crop_h * scale)))
                                resized_crop = cv2.resize(other_crop, (target_w, target_h))

                                paste_x2 = min(frame.shape[1], x2)
                                paste_y2 = min(frame.shape[0], y2)
                                paste_x1 = max(0, paste_x2 - target_w)
                                paste_y1 = max(0, paste_y2 - target_h)

                                if paste_y1 < paste_y2 and paste_x1 < paste_x2:
                                    visible_crop = resized_crop[
                                        target_h - (paste_y2 - paste_y1):,
                                        target_w - (paste_x2 - paste_x1):,
                                    ]
                                    frame[paste_y1:paste_y2, paste_x1:paste_x2] = visible_crop
                                cv2.putText(frame, 'xx', (paste_x1, max(20, paste_y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2)

                else:
                    raise Exception(f"unknown window name: {window_names[f]}")
                cv2.putText(frame, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[f], 2)

            cv2.putText(frame, f"Frame {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow(window_names[f], frame)

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
            frame_index += 1

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
