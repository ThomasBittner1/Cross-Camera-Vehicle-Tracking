# Cars Multicamera Matching

Vehicle detection, tracking, and cross-camera matching for a two-camera traffic scene.

The current pipeline treats camera 0 as the **source** camera and camera 1 as the **query** camera. Cars that leave the source camera are recorded into a temporary source gallery. Query-camera cars are compared against that gallery after crossing the configured query entry line.

## Requirements

- Python 3.10+
- CUDA-capable PyTorch install for the current TensorRT/ReID setup
- A YOLO detection model, configured in `config.py`
- Vehicle ReID checkpoint files:
  - `net_19.pth`
  - `opts.yaml`
- Input videos at the paths configured in `config.py`

Install Python packages:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python run.py
```

The app opens OpenCV windows for the source/query cameras. In normal mode, only the query window is shown, with the source camera embedded as a 30% inset in the top-left corner.

## Configuration

Main settings live in `config.py`:

- `video_paths`: source and query video paths
- `window_names`: source/query window names
- `model_path`: YOLO model path
- `start_frame_index`: first frame to process
- `debug_mode`: start with debug UI enabled
- `debug_pause_at_frame_index`: pause once at a specific frame, or `None`
- `record_to_file`: output video path for the query window, or `None`
- `source_record_ttl_seconds`: how long exited source records remain matchable
- `show_score_label`: show simplified match labels in normal mode
- `entry_line_query`: line that query cars must cross before matching is displayed
- `source_discard_lines`: source-camera lines that discard tracks instead of adding them to the gallery
- `mask_points_by_camera`: inference ignore masks
- `not_from_other_camera_masks_query_camera`: query-camera regions excluded from source matching

## Controls

- `q`: quit
- `Space`: pause/resume
- Right arrow while paused: step one frame
- Mouse click a car: isolate that track in the clicked camera; click empty space to clear
- `D`: toggle debug mode

Debug-only controls:

- `0`-`9`: number of match candidates to display
- `M`: toggle inference-ignore mask overlay
- `O`: toggle query not-from-source mask overlay
- `,` / `.`: page through source crop gallery

## Display Modes

Normal mode:

- Query camera is the main window.
- Source camera is shown as a small inset.
- Tracker labels are hidden.
- Match labels are simplified to `Very likely`, `Likely`, `Possible`, or `Weak Match` if `show_score_label` is enabled.
- Query cars show `no source found` only after crossing the entry line and failing to match.

Debug mode:

- Source and query cameras are shown in separate windows.
- Tracker IDs and crop quality labels are visible.
- Source crop gallery is shown.
- Match panels include source ID, embedding score, elapsed seconds, and strong/weak crop status.

## Main Files

- `run.py`: main application loop
- `config.py`: runtime configuration
- `yolo.py`: detection model loading and inference wrappers
- `tracking.py`: BoxMOT tracker integration
- `cross_camera_matcher.py`: source gallery and query/source matching
- `embedding_utils.py`: vehicle ReID model and embedding utilities
- `geometry_utils.py`: line, polygon, and bounding-box helpers
- `visualization.py`: OpenCV drawing, debug UI, and recording
- `track_test.py`: simpler detection/tracking visualization test
