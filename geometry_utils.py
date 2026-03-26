import numpy as np
import cv2
from collections import deque, defaultdict


def get_distributed_items(items, n=16):
    if len(items) <= n:
        return items

    indices = np.round(np.linspace(0, len(items) - 1, n)).astype(int)
    distributed_items = [items[i] for i in indices]
    return distributed_items


def is_box_overlapping(box, other_boxes, min_iou=0.2, box_id=None):
    x1, y1, x2, y2 = map(int, box[:4])
    current_area = max(0, x2 - x1) * max(0, y2 - y1)

    for other_box in other_boxes:
        other_id = int(other_box[4]) if len(other_box) > 4 else None
        if box_id is not None and other_id == box_id:
            continue

        other_x1, other_y1, other_x2, other_y2 = map(int, other_box[:4])
        inter_x1 = max(x1, other_x1)
        inter_y1 = max(y1, other_y1)
        inter_x2 = min(x2, other_x2)
        inter_y2 = min(y2, other_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h
        other_area = max(0, other_x2 - other_x1) * max(0, other_y2 - other_y1)
        union_area = current_area + other_area - intersection_area

        if union_area > 0 and (intersection_area / union_area) >= min_iou:
            return True

    return False


class TrajectoryManager:
    def __init__(self, max_points=30, max_lost_frames=5, on_delete_callback=None):
        self.max_points = max_points
        self.max_lost_frames = max_lost_frames
        self.tracks = {}
        self.on_delete_callback = on_delete_callback

    def update(self, obj_id, pos, cropped_box=None):
        if obj_id not in self.tracks:
            self.tracks[obj_id] = {
                "points": deque(maxlen=self.max_points),
                "crops": deque(maxlen=self.max_points),
                "lost_count": 0
            }

        track = self.tracks[obj_id]
        track["points"].append(pos)
        track["crops"].append(cropped_box)
        track["lost_count"] = 0

        num_points_for_distance = 3
        pts_count = len(track["points"])

        if pts_count >= 2:
            lookback = min(pts_count, num_points_for_distance)

            start_point = track["points"][-lookback]
            end_point = track["points"][-1]

            dir = np.array(end_point, dtype='float64') - np.array(start_point, dtype='float64')
            # Normalize to unit vector
            mag = np.linalg.norm(dir)
            track["mag"] = mag
            if mag > 0:
                track["direction"] = dir / mag
            else:
                track["direction"] = np.array([0.0, 0.0])
                track['angle'] = 0.0

            angle_rad = np.arctan2(dir[1], dir[0])
            angle_deg = np.degrees(angle_rad)
            track['angle'] = angle_deg

        else:
            track["direction"] = np.array([0.0, 0.0])
            track["mag"] = 0.0
            track['angle'] = 0.0

    def process_garbage_collection(self, active_ids, cid):
        # Use list() because we are deleting keys while iterating
        all_stored_ids = list(self.tracks.keys())

        for obj_id in all_stored_ids:
            track = self.tracks[obj_id]

            if obj_id not in active_ids:
                track["lost_count"] += 1
            else:
                track["lost_count"] = 0

            if track["lost_count"] > self.max_lost_frames:
                if self.on_delete_callback:
                    self.on_delete_callback(cid, obj_id, track["crops"])
                del self.tracks[obj_id]


    def draw(self, frame):
        for obj_id, data in self.tracks.items():
            points = data["points"]
            if len(points) < 2:
                continue
            dir = data['direction']
            cv2.putText(frame, f"deg:{data['angle']:.3f}",#, (mag: {data['mag']:.3f})",
                        points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for i in range(1, len(points)):
                thickness = int(2 * (i / self.max_points) + 1)
                cv2.line(frame, points[i - 1], points[i], (0, 255, 255), thickness)



def counter_clock_wise(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def segments_intersect(p1, p2, q1, q2):
    return counter_clock_wise(p1, q1, q2) != counter_clock_wise(p2, q1, q2) and \
        counter_clock_wise(p1, p2, q1) != counter_clock_wise(p1, p2, q2)

