# Cars Multicamera Matching

Re-Identifies vehicles in a traffic camera that came from another traffic camera.

There's a **source** camera and a **query** camera. Cars that leave the source camera are recorded into a 
temporary source gallery. Query-camera cars are compared against that gallery after crossing the 
query entry line.

<video autoplay muted loop controls width="1280">
    <source src="car_reid_compressed.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>


# Algorithm


# Performance
- A simple count showed 20 true positives and 6 false positives from frame 1000 to the end of the video.
This is a relatively short evaluation range, and additional testing is required to make the algorithm more robust. However, the analysis was limited by the length of the video.
- Speed: The speed most of the time varies between 8 FPS and 13 FPS. This is mostly acceptable since The camera's speed itself is 10 FPS. But it would result in skipping a few frames
if this is evaluated in realtime. Potential fixes could be converting part of the matching from python to C++, or distributing the embeddings inference of the source camera better. 

# Known Issues  
- For vehicles not visible in the source camera, he typically does not classify them as unknown, when they appear in the query camera.
There is an embedding-score threshold below which detections are classified as unknown. However, lowering this threshold too much would cause many valid matches to be discarded.

# Difficulties


# Ideas to improve
- yolo has been optimized for those videos,  



## Install

#### 1. Install Python packages:

```powershell
pip install -r requirements.txt
```
#### 2. Patch fix the boxmot package:
``` powershell
python _install_fix_boxmot.py
```


#### 3. Download Embedding Model files:
  - `net_19.pth`
  - `opts.yaml`  

https://drive.google.com/file/d/1STbsacssLtlHpUesNzuTeUPrfMlWbSKu/view  
(Source: https://github.com/regob/vehicle_mtmc)  
After downloading the file, put *net_19.pth* and *opts.yaml* into the root folder.

#### 4. Download input videos
https://www.aicitychallenge.org/2022-track1-download
Put the main folder *AICity22_Track1_MTMC_Tracking* into the root folder


## Run

```powershell
python run.py
```

## Controls
- `d`: toggle debug mode
- `q`: quit
- `Space`: pause/resume
- Right arrow while paused: step one frame
- Mouse click a car: isolate that track in the clicked camera; click empty space to clear

Debug-only controls:

- `0`-`9`: number of match candidates to display
- `M`: toggle inference-ignore mask overlay
- `O`: toggle query not-from-source mask overlay
- `,` / `.`: page through source crop gallery

## Debug Mode

- Source and query cameras are shown in separate windows.
- Source crop gallery is shown.
- Tracker IDs and crop quality labels are visible.
- Match panels include source ID, embedding score, elapsed seconds, and strong/weak crop status.

## Use of AI

AI-assisted development tools (primarily Codex) were used throughout the project to accelerate implementation, 
refactoring, and boilerplate generation.

Core algorithmic design, system integration, and debugging were performed mostly manually. 
Some utility modules, geometry helpers, and visualization code were heavily AI-assisted and subsequently reviewed/modified.