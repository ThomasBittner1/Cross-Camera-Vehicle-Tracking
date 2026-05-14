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

        self.good_crops_per_ids_source = defaultdict(list)
        self.bad_crops_per_ids_source = defaultdict(list)
        self.embeddings_per_id = {}
        self.histograms_of_crossed_source = {}
        self.embedding_histories_query = defaultdict(list)
        self.embedding_of_crossed_source_map = []
        self.embedding_of_crossed_source = np.zeros(0)
        self.comes_from_other_query_camera = {}
        self.best_matches_query = defaultdict(list)
        self.query_camera_overlap_by_track_id = {}

    def get_best_matches(self, also_show_uncrossed=False):
        if also_show_uncrossed:
            for track_id in self.best_matches_query.keys():
                self.best_matches_query[track_id].sort(key=lambda x: x["embedding_score"], reverse=True)
            return self.best_matches_query
        else:
            return_best_matches = {}
            for track_id, matches in self.best_matches_query.items():
                matches_with_elapsed_time = [
                    match for match in matches
                    if "elapsed_time_score" in match
                ]
                if matches_with_elapsed_time:
                    return_best_matches[track_id] = sorted(matches_with_elapsed_time, key=lambda x:x["global_score"], reverse=True)
            return return_best_matches


    def store_query_camera_embeddings(self, tracks, frame):
        for track in tracks:
            x1, _, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            if track_id not in self.comes_from_other_query_camera:
                bottom_center = (int((x1 + x2) / 2), y2)
                is_inside_excluded_area = any(
                    geometry_utils.point_inside_polygon(bottom_center, mask)
                    for mask in self.not_from_other_camera_masks
                )
                self.comes_from_other_query_camera[track_id] = not is_inside_excluded_area

        non_overlapping_crops = []
        non_overlapping_track_ids = []
        self.query_camera_overlap_by_track_id = {}

        for track in tracks:
            track_id = int(track[4])
            if not self.comes_from_other_query_camera[track_id]:
                self.query_camera_overlap_by_track_id[track_id] = None
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
            self.query_camera_overlap_by_track_id[track_id] = is_overlapping
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
            self.embedding_histories_query[track_id].append(vector)

    def query_camera_track_is_relevant(self, track_id):
        return self.comes_from_other_query_camera.get(track_id, True)

    def record_source_camera_crop(self, track, tracks, frame):
        track_id = int(track[4])
        x1, y1, x2, y2 = map(int, track[:4])
        is_overlapping = geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
        width = x2 - x1
        crop = geometry_utils.get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8)

        if not is_overlapping and width > 90:
            self.good_crops_per_ids_source[track_id].append(crop)
            return True

        self.bad_crops_per_ids_source[track_id].append(crop)
        return False

    def record_embeddings(self, track_id):
        crops = self.good_crops_per_ids_source[track_id] or self.bad_crops_per_ids_source[track_id]
        embedding = calculate_embedding_multiple(self.embedder, crops)
        if embedding is None:
            return

        self.embeddings_per_id[track_id] = embedding
        self.histograms_of_crossed_source[track_id] = color_utils.calculate_histograms_multiple(crops)

    def refresh_source_camera_gallery(self):
        self.embedding_of_crossed_source = np.zeros((len(self.embeddings_per_id), self.embedding_size), dtype="float64")
        self.embedding_of_crossed_source_map.clear()

        for index, other_track_id in enumerate(sorted(self.embeddings_per_id.keys())):
            self.embedding_of_crossed_source[index] = self.embeddings_per_id[other_track_id]
            self.embedding_of_crossed_source_map.append(other_track_id)

    def update_query_camera_matches(self, track_id, crossed_times_by_camera):
        is_overlapping = self.query_camera_overlap_by_track_id.get(track_id)
        if is_overlapping or not self.embedding_histories_query[track_id]:
            return

        query_embedding = np.mean(self.embedding_histories_query[track_id], axis=0)
        if self.embedding_of_crossed_source.size == 0 or not self.embedding_of_crossed_source_map:
            return

        closest_indices, embedding_scores = embedding_utils.find_closest_embeddings(
            query_embedding,
            self.embedding_of_crossed_source,
        )
        if not closest_indices:
            return

        for closest_index, embedding_score in zip(closest_indices, embedding_scores):
            other_track_id = self.embedding_of_crossed_source_map[closest_index]
            other_draw_crop = self._best_crop_for_source_camera_track(other_track_id)

            elapsed_time = -1.0
            if track_id in crossed_times_by_camera[1]:
                elapsed_time = crossed_times_by_camera[1][track_id] - crossed_times_by_camera[0][other_track_id]

            match_got_updated = False
            for match_data in self.best_matches_query[track_id]:
                if match_data["other_track_id"] == other_track_id:
                    match_data["embedding_score"] = embedding_score
                    if elapsed_time != -1.0:
                        match_data["elapsed_time"] = elapsed_time
                    match_got_updated = True
                    self._recalculate_scores(match_data)
                    break

            if not match_got_updated:
                match_data = {
                    "embedding_score": embedding_score,
                    "other_draw_crop": other_draw_crop,
                    "other_track_id": other_track_id,
                    "elapsed_time": elapsed_time,
                }
                self._recalculate_scores(match_data)
                self.best_matches_query[track_id].append(match_data)

    def update_query_camera_elapsed_times(self, track_id, crossed_times_by_camera):
        if track_id not in self.best_matches_query or track_id not in crossed_times_by_camera[1]:
            return

        for match_data in self.best_matches_query[track_id]:
            if match_data["elapsed_time"] == -1.0:
                other_track_id = match_data["other_track_id"]
                elapsed_time = crossed_times_by_camera[1][track_id] - crossed_times_by_camera[0][other_track_id]
                match_data["elapsed_time"] = elapsed_time
                self._recalculate_scores(match_data)


    def _recalculate_scores(self, match_data):
        elapsed_time = match_data.get("elapsed_time", -1.0)
        if elapsed_time != -1.0:
            match_data["elapsed_time_score"] = np.interp(
                elapsed_time,
                [0, 40, 50, 65, 75],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            )
        match_data["global_score"] = match_data["embedding_score"] * match_data.get("elapsed_time_score", 1.0)


    def _best_crop_for_source_camera_track(self, track_id):
        if self.good_crops_per_ids_source[track_id]:
            self.good_crops_per_ids_source[track_id].sort(key=lambda crop: crop.shape[1])
            return self.good_crops_per_ids_source[track_id][-1]

        self.bad_crops_per_ids_source[track_id].sort(key=lambda crop: crop.shape[1])
        return self.bad_crops_per_ids_source[track_id][-1]
