# Overwatch 2 Scoreboard Parser

A computer vision tool for extracting player statistics from Overwatch 2 replay viewer screenshots.

## Overview

This parser processes screenshots of the Overwatch 2 in-game scoreboard (accessible via Tab during replay viewing) and extracts structured data for all 10 players including roles, names, and statistics.

## Features

### Current Capabilities

- **Dynamic scoreboard detection**: Automatically detects scoreboard boundaries using gradient-based edge detection. Handles variable scoreboard widths caused by different perk configurations.
- **Role recognition**: Identifies player roles (tank, dps, support) using saturation-based template matching that works across both team colors (blue/yellow).
- **Player name OCR**: Extracts player names using PaddleOCR with preprocessing optimized for white text on colored backgrounds.
- **Statistics OCR**: Extracts all visible stats: eliminations, assists, deaths, damage, healing, and mitigation.

### Planned Features

- Hero portrait recognition
- Ultimate status detection (ready checkmark vs charge percentage)
- Automated screenshot capture harness

## Technical Implementation

### Edge Detection

The scoreboard width varies based on the number of perks equipped by heroes. The parser scans horizontally at the expected scoreboard height and detects sharp brightness gradients to find the left and right edges.

### Column Position Calculation

Column positions are calculated dynamically from detected edges using known column widths. The layout consists of:
- Left section: role icon, hero portrait, ult status (fixed widths)
- Center section: player name, report button (variable width)
- Right section: statistics columns (fixed widths)

### Template Matching

Role icons are matched using HSV saturation extraction. White icons have low saturation regardless of the background color (blue for team 1, yellow for team 2), allowing a single set of templates to work for both teams.

### OCR Pipeline

Text recognition uses PaddleOCR with custom preprocessing:
1. 3x upscaling for better accuracy on small crops
2. HSV saturation thresholding to isolate white text
3. Morphological erosion to thicken thin characters

## Project Structure

```
scoreboard_parser/
├── src/
│   ├── recognize.py    # Recognition module (roles, OCR, frame processing)
│   ├── regions.py      # Layout constants and column widths
│   ├── utils.py        # Edge detection, cropping utilities
│   ├── capture.py      # Screenshot capture (planned)
│   └── process.py      # Batch processing (planned)
├── debug/
│   ├── test_frame_processing.py  # Full pipeline test
│   ├── test_ocr.py               # OCR testing
│   ├── test_role_matching.py     # Role template matching test
│   ├── test_preprocessing.py     # OCR preprocessing visualization
│   ├── show_columns.py           # Column boundary visualization
│   └── show_edges.py             # Edge detection visualization
├── templates/
│   └── roles/          # Role icon templates (tank.png, dps.png, support.png)
├── screenshots/        # Input screenshots
└── output/             # Debug output images
```

## Installation

Requires Python 3.12+.

```bash
# Clone the repository
git clone <repository-url>
cd scoreboard_parser

# Install dependencies using uv
uv sync

# Install PaddlePaddle (required for OCR)
uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

## Usage

### Process a Screenshot

```bash
uv run python debug/test_frame_processing.py screenshots/1.png
```

Output:
```
================================================================================
FRAME PROCESSING RESULTS
================================================================================

Team 1 (Blue)
----------------------------------------
  [tank   ] PLAYERNAME      | E:4 | A:1 | D:1 | Dmg:5214 | Heal:0 | Mit:3784
  ...

Team 2 (Yellow)
----------------------------------------
  [support] PLAYERNAME      | E:1 | A:2 | D:0 | Dmg:690 | Heal:5887 | Mit:0
  ...
```

### Programmatic Usage

```python
from src.recognize import process_frame
from src.utils import load_image

image = load_image("screenshots/1.png")
players = process_frame(image)

for player in players:
    print(f"{player.name}: {player.role}, E:{player.elims}, D:{player.deaths}")
```

### Debug Tools

```bash
# Visualize detected scoreboard edges
uv run python debug/show_edges.py

# Visualize column boundaries
uv run python debug/show_columns.py

# Test role template matching
uv run python debug/test_role_matching.py

# Visualize OCR preprocessing at different thresholds
uv run python debug/test_preprocessing.py
```

## Dependencies

- `opencv-python`: Image processing and template matching
- `paddleocr`: Text recognition
- `paddlepaddle`: PaddleOCR backend (CPU version)
- `mss`: Screen capture
- `pillow`: Image loading
- `pandas`: Data export (planned)

## Known Limitations

- OCR may occasionally misread thin italic characters (e.g., "I" at word boundaries)
- Large numbers with similar-looking digits (3/8, 5/6) may be misread in rare cases
- Requires screenshots at native resolution for accurate column detection

## License

MIT