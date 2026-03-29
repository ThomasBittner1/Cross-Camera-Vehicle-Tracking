import numpy as np
import cv2
from collections import deque, defaultdict



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


def get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8):
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    shrunk_w = max(1, int(round(box_w * scale)))
    shrunk_h = max(1, int(round(box_h * scale)))

    shrunk_x1 = max(0, int(round(center_x - shrunk_w / 2.0)))
    shrunk_y1 = max(0, int(round(center_y - shrunk_h / 2.0)))
    shrunk_x2 = min(frame.shape[1], shrunk_x1 + shrunk_w)
    shrunk_y2 = min(frame.shape[0], shrunk_y1 + shrunk_h)

    return frame[shrunk_y1:shrunk_y2, shrunk_x1:shrunk_x2]


def point_inside_box(point, box_coords):
    px, py = point
    x1, y1, x2, y2 = box_coords
    return x1 <= px <= x2 and y1 <= py <= y2



def counter_clock_wise(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def segments_intersect(p1, p2, q1, q2):
    return counter_clock_wise(p1, q1, q2) != counter_clock_wise(p2, q1, q2) and \
        counter_clock_wise(p1, p2, q1) != counter_clock_wise(p1, p2, q2)

