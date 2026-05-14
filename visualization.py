import cv2
import numpy as np


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.num_other_matches_to_show = 5
        self.debug_matches = False
        self.show_inference_ignore_area = False
        self.show_not_from_other_camera_area = False

    def handle_key(self, key):
        if ord("0") <= key <= ord("9"):
            self.num_other_matches_to_show = key - ord("0")
        elif key in (ord("d"), ord("D")):
            self.debug_matches = not self.debug_matches
        elif key in (ord("m"), ord("M")):
            self.show_inference_ignore_area = not self.show_inference_ignore_area
        elif key in (ord("o"), ord("O")):
            self.show_not_from_other_camera_area = not self.show_not_from_other_camera_area

    def draw(self, original_frames, frame_draw_data_by_camera, isolated_track_id_by_camera, best_matches_1):
        for camera_index in [0, 1]:
            draw_frame = original_frames[camera_index].copy()
            draw_data = frame_draw_data_by_camera[camera_index]
            isolated_track_id = isolated_track_id_by_camera[camera_index]

            self._draw_overlays(camera_index, draw_frame)
            if draw_data["line"] is not None:
                cv2.line(draw_frame, draw_data["line"][0], draw_data["line"][1], (0, 0, 255), 2)
            for exit_line in draw_data["disappear_lines"]:
                cv2.line(draw_frame, exit_line[0], exit_line[1], (0, 255, 255), 2)

            for box in draw_data["boxes"]:
                if isolated_track_id is not None and box["track_id"] != isolated_track_id:
                    continue

                x1, y1, x2, _ = box["coords"]
                cv2.rectangle(draw_frame, (x1, y1), (x2, box["coords"][3]), box["box_color"], 2)
                cv2.putText(
                    draw_frame,
                    box["label"],
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box["label_color"],
                    2,
                )

                if camera_index == 1 and box["track_id"] in best_matches_1:
                    self._draw_match_panel(draw_frame, box, best_matches_1[box["track_id"]])

            self._draw_legend(draw_frame, draw_data["frame_text"], draw_data["fps_text"])
            cv2.imshow(self.config.window_names[camera_index], draw_frame)

    def _draw_overlays(self, camera_index, draw_frame):
        display = self.config.display
        if self.show_inference_ignore_area:
            overlay = draw_frame.copy()
            cv2.fillPoly(
                overlay,
                [np.array(self.config.mask_points_by_camera[camera_index], dtype=np.int32)],
                display.inference_ignore_area_color,
            )
            cv2.addWeighted(
                overlay,
                display.not_from_other_camera_area_alpha,
                draw_frame,
                1 - display.not_from_other_camera_area_alpha,
                0,
                draw_frame,
            )

        if camera_index == 1 and self.show_not_from_other_camera_area:
            overlay = draw_frame.copy()
            for mask_points in self.config.not_from_other_camera_masks_query_camera:
                cv2.fillPoly(
                    overlay,
                    [np.array(mask_points, dtype=np.int32)],
                    display.not_from_other_camera_area_color,
                )
            cv2.addWeighted(
                overlay,
                display.inference_ignore_area_alpha,
                draw_frame,
                1 - display.inference_ignore_area_alpha,
                0,
                draw_frame,
            )

    def _draw_match_panel(self, draw_frame, box, matches):
        matches_to_draw = self._visible_matches(matches)
        if not matches_to_draw:
            self._draw_no_matches_found(draw_frame, box)
            return

        x1, _, x2, y2 = box["coords"]
        panel_items = []
        panel_width = 0
        panel_height = 0

        for match_data in reversed(matches_to_draw):
            other_draw_crop = match_data["other_draw_crop"]
            other_label = (
                f"{match_data['other_track_id']} "
                f"score: {match_data['embedding_score']:.2f} / {match_data['elapsed_ms_score']:.1f} "
                f"({match_data['elapsed_ms']:.1f}) / {match_data['global_score']:.2f}\n"
                f"{'strong' if match_data['is_strong'] else 'weak'}")

            crop_h, crop_w = other_draw_crop.shape[:2]
            box_w = max(1, x2 - x1)
            target_w = max(1, int(round(box_w * 0.5)))
            scale = target_w / max(1, crop_w)
            target_h = max(1, int(round(crop_h * scale)))

            panel_items.append(
                {
                    "crop": other_draw_crop,
                    "target_w": target_w,
                    "target_h": target_h,
                    "label": other_label,
                }
            )
            panel_width = max(panel_width, target_w)
            panel_height += target_h

        if panel_width == 0 or panel_height == 0:
            return

        panel = np.zeros((panel_height, panel_width, 3), dtype=draw_frame.dtype)
        text_items = []
        panel_y = 0
        for item in panel_items:
            resized_crop = cv2.resize(item["crop"], (item["target_w"], item["target_h"]))
            panel_x1 = panel_width - item["target_w"]
            panel_y1 = panel_y
            panel_y2 = panel_y + item["target_h"]
            panel[panel_y1:panel_y2, panel_x1:panel_width] = resized_crop
            text_items.append({"label": item["label"], "x": panel_x1, "y": panel_y2 - 3})
            panel_y = panel_y2

        frame_h, frame_w = draw_frame.shape[:2]
        paste_x2 = min(frame_w, x2)
        paste_y2 = min(frame_h, y2)
        paste_x1 = max(0, paste_x2 - panel_width)
        paste_y1 = max(0, paste_y2 - panel_height)
        visible_w = paste_x2 - paste_x1
        visible_h = paste_y2 - paste_y1
        if visible_w <= 0 or visible_h <= 0:
            return

        visible_panel = panel[panel_height - visible_h:, panel_width - visible_w:]
        draw_frame[paste_y1:paste_y2, paste_x1:paste_x2] = visible_panel

        hidden_x = panel_width - visible_w
        hidden_y = panel_height - visible_h
        for text_item in text_items:
            text_x = text_item["x"] - hidden_x
            text_y = text_item["y"] - hidden_y
            if text_y < 0 or text_y >= visible_h:
                continue
            cv2.putText(
                draw_frame,
                text_item["label"],
                (paste_x1 + max(0, text_x), paste_y1 + text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.display.colors_by_camera[0],
                2,
            )

    def _visible_matches(self, matches):
        if self.debug_matches:
            return matches[0:self.num_other_matches_to_show]

        if matches and matches[0]["global_score"] > 0.0:
            return matches[0:1]

        return []

    def _draw_no_matches_found(self, draw_frame, box):
        x1, _, x2, y2 = box["coords"]
        label = "no matches found"
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        frame_h, frame_w = draw_frame.shape[:2]
        x = min(max(0, x1), max(0, frame_w - text_w - 4))
        y = min(max(text_h + baseline + 4, y2), frame_h - baseline - 4)
        cv2.putText(
            draw_frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.config.display.colors_by_camera[0],
            2,
        )

    def _draw_legend(self, draw_frame, frame_text, fps_text):
        legend_lines = [
            frame_text,
            fps_text,
            (
                f"D: debug matches ({'on' if self.debug_matches else 'off'})  "
                f"0-9: matches ({self.num_other_matches_to_show})  "
                f"M: inference-ignore ({'on' if self.show_inference_ignore_area else 'off'})  "
                f"O: not-from-other-camera ({'on' if self.show_not_from_other_camera_area else 'off'})"
            ),
        ]
        for line_idx, legend_line in enumerate(legend_lines):
            y = 30 + line_idx * 28
            cv2.putText(draw_frame, legend_line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
