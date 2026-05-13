from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DisplayConfig:
    colors_by_camera: tuple[tuple[int, int, int], tuple[int, int, int]] = ((255, 0, 0), (0, 255, 0))
    inference_ignore_area_color: tuple[int, int, int] = (255, 0, 0)
    inference_ignore_area_alpha: float = 0.5
    not_from_other_camera_area_color: tuple[int, int, int] = (0, 0, 255)
    not_from_other_camera_area_alpha: float = 0.5


@dataclass(frozen=True)
class AppConfig:
    start_frame_index: int = 1000
    # model_path: Path = Path(r"C:\ComputerVision\_Datasets_\tb_dataManager\runs_cars_multicamera\train\weights\best.pt")
    model_path: Path = Path(r"C:\ComputerVision\_Datasets_\tb_dataManager\runs_cars_multicamera\train2\weights\best.engine")
    window_names: tuple[str, str] = ("c042", "c041")
    video_paths: tuple[str, str] = (
        r"AICity22_Track1_MTMC_Tracking\test\S06\c042\vdo.avi",
        r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi",
    )
    cross_lines: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = (
        ((773, 175), (953, 256)),
        ((227, 283), (731, 956)),
    )
    mask_points_by_camera: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = (
        (
            (1278, 493), (961, 256), (1101, 163), (1027, 101), (881, 126), (684, 165),
            (499, 128), (304, 142), (168, 145), (7, 222), (57, 290), (2, 352),
            (0, 7), (1278, 4),
        ),
        (
            (1, 293), (146, 205), (105, 124), (247, 86), (424, 135), (534, 116),
            (728, 168), (1011, 148), (1133, 145), (1199, 197), (1087, 273),
            (1138, 361), (1278, 412), (1276, 3), (5, 3),
        ),
    )
    not_from_other_camera_masks_query_camera: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = (
        ((657, 948), (1083, 286), (1278, 419), (1277, 956)),
        ((2, 370), (536, 188), (888, 162), (1277, 199), (1275, 5), (2, 4)),
    )
    display: DisplayConfig = DisplayConfig()
