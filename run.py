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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c042_cross_line = [(773, 175), (953, 256)]
c041_cross_line = [(260, 331), (802, 906)]
# masks are created with draw_mask.py

mask_c042 = [(0, 416), (721, 147), (963, 122), (1074, 197), (244, 959), (1, 955)]
mask_c041 = [(4, 392), (336, 269), (766, 180), (1033, 160), (1144, 238), (556, 912), (334, 958), (5, 959)]

c042_other_best_crops = {}
c042_other_best_embedding_distance = {}

EMBEDDING_SIZE = 2048

def calculate_embedding_exited_car(crops):
    distributed_crops = geometry_utils.get_distributed_items(crops)
    vector = embedder.get_embeddings(distributed_crops)
    mean_vector = np.mean(vector, axis=0)
    return mean_vector
    print ('mean_vector: ', mean_vector)



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
    embedding_vectors_of_crossed_c041 = {}

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

            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crops_per_ids[f][track_id].append(frame[y1:y2, x1:x2])
                label = f"ID {track_id}"

                if window_names[f] == "c042":
                    cv2.line(frame, c042_cross_line[0], c042_cross_line[1], (0, 0, 255), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    prev = prev_centers[f].get(track_id)
                    if prev and track_id not in crossed_ids[f]:
                        if geometry_utils.segments_intersect(prev, (cx, cy), c042_cross_line[0], c042_cross_line[1]):
                            crossed_ids[f].add(track_id)
                            embedding_vectors_of_crossed_c042[track_id] = calculate_embedding_exited_car(crops_per_ids[f][track_id])

                    prev_centers[f][track_id] = (cx, cy)
                    if track_id in crossed_ids[f]:
                        label = f"{label} crossed"
                elif window_names[f] == "c041":

                    cv2.line(frame, c041_cross_line[0], c041_cross_line[1], (0, 0, 255), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    prev = prev_centers[f].get(track_id)
                    if prev and track_id not in crossed_ids[f]:

                        if geometry_utils.segments_intersect(prev, (cx, cy), c041_cross_line[0], c041_cross_line[1]):
                            print(f'crossed in c041 - {track_id}')
                            query_embedding = calculate_embedding_exited_car(crops_per_ids[f][track_id])

                            crossed_ids[f].add(track_id)

                            gallery = np.zeros((len(embedding_vectors_of_crossed_c042), EMBEDDING_SIZE), dtype='float64')
                            gallery_map = []
                            for t, other_track_id in enumerate(sorted(embedding_vectors_of_crossed_c042.keys())):
                                gallery[t] = embedding_vectors_of_crossed_c042[other_track_id]
                                gallery_map.append(other_track_id)
                            closest_embedding_idx, closest_embedding_score = embedding_utils.find_closest_embedding(query_embedding, gallery)
                            other_track_id = gallery_map[closest_embedding_idx]
                            widths = [x.shape[1] for x in crops_per_ids[0][other_track_id]]
                            biggest_shape_idx = np.argmax(widths)
                            c042_other_best_crops[track_id] = crops_per_ids[0][other_track_id][biggest_shape_idx]
                            c042_other_best_embedding_distance[track_id] = closest_embedding_score
                    prev_centers[f][track_id] = (cx, cy)
                    if track_id in crossed_ids[f]:
                        label = f"{label} crossed"

                    if track_id in c042_other_best_crops:
                        other_crop = c042_other_best_crops[track_id]
                        crop_h, crop_w = other_crop.shape[:2]
                        paste_y1 = y2
                        paste_y2 = min(frame.shape[0], paste_y1 + crop_h)
                        paste_x1 = max(0, x1)
                        paste_x2 = min(frame.shape[1], paste_x1 + crop_w)

                        if paste_y1 < frame.shape[0] and paste_x1 < frame.shape[1]:
                            visible_crop = other_crop[:paste_y2 - paste_y1, :paste_x2 - paste_x1]
                            frame[paste_y1:paste_y2, paste_x1:paste_x2] = visible_crop
                            cv2.putText(frame, f"score: {c042_other_best_embedding_distance[track_id]}",
                                        (paste_x1, min(frame.shape[0] - 10, paste_y2 + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    raise Exception(f"unknown window name: {window_names[f]}")
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
