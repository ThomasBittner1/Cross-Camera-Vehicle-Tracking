from collections import defaultdict

import numpy as np

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

        self.strong_crops_per_ids_source = defaultdict(list)
        self.weak_crops_per_ids_source = defaultdict(list)
        self.embeddings_per_id = {}
        self.embedding_histories_query = defaultdict(list)
        self.embedding_of_exited_source_map = []
        self.embedding_of_exited_source = np.zeros(0)
        self.query_comes_from_source = {}
        self.best_matches_query = defaultdict(dict)
        self.query_camera_overlap_by_track_id = {}

    def get_best_matches(self):
        return {track_id: sorted(matches_by_source_id.values(), key=lambda match: match["global_score"], reverse=True)
                for track_id, matches_by_source_id in self.best_matches_query.items() if matches_by_source_id}


    def process_query_embeddings(self, tracks, frame):
        for track in tracks:
            x1, _, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            if track_id not in self.query_comes_from_source:
                bottom_center = (int((x1 + x2) / 2), y2)
                is_inside_excluded_area = any(
                    geometry_utils.point_inside_polygon(bottom_center, mask)
                    for mask in self.not_from_other_camera_masks)
                self.query_comes_from_source[track_id] = not is_inside_excluded_area

        non_overlapping_crops = []
        non_overlapping_track_ids = []
        self.query_camera_overlap_by_track_id = {}

        for track in tracks:
            track_id = int(track[4])
            if not self.query_comes_from_source[track_id]:
                self.query_camera_overlap_by_track_id[track_id] = None
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            is_overlapping = False #geometry_utils.is_box_overlapping(track, tracks, min_iou=0.1, box_id=track_id)
            self.query_camera_overlap_by_track_id[track_id] = is_overlapping
            if not is_overlapping:
                non_overlapping_track_ids.append(track_id)
                non_overlapping_crops.append(geometry_utils.get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8))

        all_visible_query_embeddings = calculate_embedding_multiple(self.embedder,
                                                                    non_overlapping_crops,
                                                                    distributed_count=None,
                                                                    return_mean=False)
        if all_visible_query_embeddings is None:
            return

        for index, vector in enumerate(all_visible_query_embeddings):
            track_id = non_overlapping_track_ids[index]
            self.embedding_histories_query[track_id].append(vector)

    def query_camera_track_is_relevant(self, track_id):
        return self.query_comes_from_source.get(track_id, True)

    def append_source_camera_crop(self, track_id, crop, is_strong_crop):
        if track_id == 26:
            print ('adding 26!!!!')
        if is_strong_crop:
            self.strong_crops_per_ids_source[track_id].append(crop)
        else:
            self.weak_crops_per_ids_source[track_id].append(crop)

    def discard_source_camera_track(self, track_id):
        gallery_needs_refresh = track_id in self.embeddings_per_id

        self.strong_crops_per_ids_source.pop(track_id, None)
        self.weak_crops_per_ids_source.pop(track_id, None)
        self.embeddings_per_id.pop(track_id, None)

        # for matches_by_source_id in self.best_matches_query.values():
        #     matches_by_source_id.pop(track_id, None)

        if gallery_needs_refresh:
            self.refresh_source_camera_gallery()

    def record_embeddings(self, track_id):
        crops = self.strong_crops_per_ids_source[track_id] or self.weak_crops_per_ids_source[track_id]
        embedding = calculate_embedding_multiple(self.embedder, crops)
        if embedding is None:
            return

        self.embeddings_per_id[track_id] = embedding

    def refresh_source_camera_gallery(self):
        self.embedding_of_exited_source = np.zeros((len(self.embeddings_per_id), self.embedding_size), dtype="float64")
        self.embedding_of_exited_source_map.clear()

        for index, other_track_id in enumerate(sorted(self.embeddings_per_id.keys())):
            self.embedding_of_exited_source[index] = self.embeddings_per_id[other_track_id]
            self.embedding_of_exited_source_map.append(other_track_id)

    def check_matches(self, track_id, crossed_time, exited_times_source):

        is_overlapping = self.query_camera_overlap_by_track_id.get(track_id)
        if is_overlapping or not self.embedding_histories_query[track_id]:
            return

        query_embedding = np.mean(self.embedding_histories_query[track_id], axis=0)
        if self.embedding_of_exited_source.size == 0 or not self.embedding_of_exited_source_map:
            return

        closest_indices, embedding_scores = embedding_utils.find_closest_embeddings(
            query_embedding,
            self.embedding_of_exited_source)

        if not closest_indices:
            return

        for closest_index, embedding_score in zip(closest_indices, embedding_scores):
            other_track_id = self.embedding_of_exited_source_map[closest_index]
            is_strong, other_draw_crop = self._best_crop_for_source_camera_track(other_track_id)

            elapsed_ms = crossed_time - exited_times_source[other_track_id]
            if elapsed_ms < 15.0:
                continue

            match_data = self.best_matches_query[track_id].get(other_track_id)
            if match_data is None:
                match_data = {
                    "embedding_score": embedding_score,
                    "other_draw_crop": other_draw_crop,
                    "other_track_id": other_track_id,
                    "elapsed_ms": elapsed_ms,
                }
                self.best_matches_query[track_id][other_track_id] = match_data
            else:
                match_data["embedding_score"] = embedding_score
                match_data["elapsed_ms"] = elapsed_ms

            match_data["is_strong"] = is_strong

            if not is_strong:
                match_data["embedding_score"] *= 1.1

            elapsed_ms_score = 0.0 if match_data['elapsed_ms'] < 15.0 or match_data['elapsed_ms'] > 60.0 else 1.0
            match_data["elapsed_ms_score"] = elapsed_ms_score
            if match_data["embedding_score"] < 0.35:
                match_data["embedding_score"] = 0.0
            match_data["global_score"] = match_data["embedding_score"] * match_data["elapsed_ms_score"]



    def _best_crop_for_source_camera_track(self, track_id):
        if self.strong_crops_per_ids_source[track_id]:
            self.strong_crops_per_ids_source[track_id].sort(key=lambda crop: crop.shape[1])
            return True, self.strong_crops_per_ids_source[track_id][-1]

        self.weak_crops_per_ids_source[track_id].sort(key=lambda crop: crop.shape[1])
        return False, self.weak_crops_per_ids_source[track_id][-1]
