from collections import defaultdict

import numpy as np

import color_utils
import embedding_utils
import general_utils
import geometry_utils


def calculate_embedding_multiple(embedder, crops, distributed_count=16, return_mean=True):
    if distributed_count:
        distributed_crops = general_utils.get_distributed_items(crops, n=distributed_count)
    else:
        distributed_crops = crops

    vectors = embedder.get_embeddings(distributed_crops)
    if vectors is None:
        return None
    if return_mean:
        return np.mean(vectors, axis=0)
    return vectors


class CrossCameraMatcher:
    def __init__(self, embedder, not_from_other_camera_masks):
        self.embedder = embedder
        self.embedding_size = embedder.embedding_dim
        self.not_from_other_camera_masks = not_from_other_camera_masks

        self.good_crops_per_ids_0 = defaultdict(list)
        self.bad_crops_per_ids_0 = defaultdict(list)
        self.embeddings_of_crossed_per_id_0 = {}
        self.histograms_of_crossed_0 = {}
        self.embedding_histories_1 = defaultdict(list)
        self.embedding_of_crossed_0_map = []
        self.embedding_of_crossed_0 = np.zeros(0)
        self.comes_from_other_camera_1 = {}
        self.best_matches_1 = defaultdict(list)
        self.camera_1_overlap_by_track_id = {}

    def store_camera_1_embeddings(self, tracks, frame):
        for track in tracks:
            x1, _, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            if track_id not in self.comes_from_other_camera_1:
                bottom_center = (int((x1 + x2) / 2), y2)
                is_inside_excluded_area = any(
                    geometry_utils.point_inside_polygon(bottom_center, mask)
                    for mask in self.not_from_other_camera_masks
                )
                self.comes_from_other_camera_1[track_id] = not is_inside_excluded_area

        non_overlapping_crops = []
        non_overlapping_track_ids = []
        self.camera_1_overlap_by_track_id = {}

        for track in tracks:
            track_id = int(track[4])
            if not self.comes_from_other_camera_1[track_id]:
                self.camera_1_overlap_by_track_id[track_id] = None
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
            self.camera_1_overlap_by_track_id[track_id] = is_overlapping
            if not is_overlapping:
                non_overlapping_track_ids.append(track_id)
                non_overlapping_crops.append(
                    geometry_utils.get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8)
                )

        current_embeddings = calculate_embedding_multiple(
            self.embedder,
            non_overlapping_crops,
            distributed_count=None,
            return_mean=False,
        )
        if current_embeddings is None:
            return

        for index, vector in enumerate(current_embeddings):
            track_id = non_overlapping_track_ids[index]
            self.embedding_histories_1[track_id].append(vector)

    def camera_1_track_is_relevant(self, track_id):
        return self.comes_from_other_camera_1.get(track_id, True)

    def record_camera_0_crop(self, track, tracks, frame):
        track_id = int(track[4])
        x1, y1, x2, y2 = map(int, track[:4])
        is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
        width = x2 - x1
        crop = geometry_utils.get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8)

        if not is_overlapping and width > 90:
            self.good_crops_per_ids_0[track_id].append(crop)
            return True

        self.bad_crops_per_ids_0[track_id].append(crop)
        return False

    def record_camera_0_crossing(self, track_id):
        crops = self.good_crops_per_ids_0[track_id] or self.bad_crops_per_ids_0[track_id]
        embedding = calculate_embedding_multiple(self.embedder, crops)
        if embedding is None:
            return

        self.embeddings_of_crossed_per_id_0[track_id] = embedding
        self.histograms_of_crossed_0[track_id] = color_utils.calculate_histograms_multiple(crops)

    def refresh_camera_0_gallery(self):
        self.embedding_of_crossed_0 = np.zeros((len(self.embeddings_of_crossed_per_id_0), self.embedding_size), dtype="float64")
        self.embedding_of_crossed_0_map.clear()

        for index, other_track_id in enumerate(sorted(self.embeddings_of_crossed_per_id_0.keys())):
            self.embedding_of_crossed_0[index] = self.embeddings_of_crossed_per_id_0[other_track_id]
            self.embedding_of_crossed_0_map.append(other_track_id)

    def update_camera_1_matches(self, track_id, crossed_times_by_camera):
        is_overlapping = self.camera_1_overlap_by_track_id.get(track_id)
        if is_overlapping or not self.embedding_histories_1[track_id]:
            return

        query_embedding = np.mean(self.embedding_histories_1[track_id], axis=0)
        if self.embedding_of_crossed_0.size == 0 or not self.embedding_of_crossed_0_map:
            return

        closest_indices, embedding_scores = embedding_utils.find_closest_embeddings(
            query_embedding,
            self.embedding_of_crossed_0,
        )
        if not closest_indices:
            return

        for closest_index, embedding_score in zip(closest_indices, embedding_scores):
            other_track_id = self.embedding_of_crossed_0_map[closest_index]
            other_draw_crop = self._best_crop_for_camera_0_track(other_track_id)

            elapsed_time = -1.0
            if track_id in crossed_times_by_camera[1]:
                elapsed_time = crossed_times_by_camera[1][track_id] - crossed_times_by_camera[0][other_track_id]

            self._upsert_match(track_id, other_track_id, embedding_score, other_draw_crop, elapsed_time)

    def update_camera_1_elapsed_times(self, track_id, crossed_times_by_camera):
        if track_id not in self.best_matches_1 or track_id not in crossed_times_by_camera[1]:
            return

        for match_data in self.best_matches_1[track_id]:
            if match_data["elapsed_time"] == -1.0:
                other_track_id = match_data["other_track_id"]
                match_data["elapsed_time"] = crossed_times_by_camera[1][track_id] - crossed_times_by_camera[0][other_track_id]

        self.best_matches_1[track_id].sort(key=lambda x: x["embedding_score"], reverse=True)

    def _best_crop_for_camera_0_track(self, track_id):
        if self.good_crops_per_ids_0[track_id]:
            self.good_crops_per_ids_0[track_id].sort(key=lambda crop: crop.shape[1])
            return self.good_crops_per_ids_0[track_id][-1]

        self.bad_crops_per_ids_0[track_id].sort(key=lambda crop: crop.shape[1])
        return self.bad_crops_per_ids_0[track_id][-1]

    def _upsert_match(self, track_id, other_track_id, embedding_score, other_draw_crop, elapsed_time):
        for match_data in self.best_matches_1[track_id]:
            if match_data["other_track_id"] == other_track_id:
                match_data["embedding_score"] = embedding_score
                if elapsed_time != -1.0:
                    match_data["elapsed_time"] = elapsed_time
                return

        self.best_matches_1[track_id].append(
            {
                "embedding_score": embedding_score,
                "other_draw_crop": other_draw_crop,
                "other_track_id": other_track_id,
                "elapsed_time": elapsed_time,
            }
        )
