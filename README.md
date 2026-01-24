# Overwatch 2 Scoreboard Parser

A computer vision tool for extracting player statistics from Overwatch 2 replay viewer screenshots.

## Overview

This parser processes screenshots of the Overwatch 2 in-game scoreboard (accessible via Tab during replay viewing) and extracts structured data for all 10 players. The extracted data includes player roles, hero selections, names, ultimate status, and all visible statistics.

## Features

- **Dynamic scoreboard detection**: Automatically detects scoreboard boundaries using gradient-based edge detection. Handles variable scoreboard widths caused by different perk configurations.

- **Role recognition**: Identifies player roles (tank, dps, support) using saturation-based template matching that works across both team colors (cyan/yellow).

- **Hero recognition**: Matches hero portraits against a library of templates with confidence thresholds to handle dead/greyed-out states.

- **Ultimate status detection**: Detects whether ultimate is ready (checkmark icon) or charging (reads percentage via OCR).

- **Player name OCR**: Extracts player names using PaddleOCR with preprocessing optimized for white text on colored backgrounds.

- **Statistics extraction**: Uses digit template matching to extract eliminations, assists, deaths, damage, healing, and mitigation values.

- **Data export**: Outputs structured data as CSV or Parquet files via pandas.

## Requirements

- Python 3.12 or higher
- Windows, macOS, or Linux

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd scoreboard_parser

# Install dependencies using uv
uv sync
```

### PaddlePaddle Installation

PaddleOCR requires PaddlePaddle as a backend. The package is included in dependencies, but if you encounter issues:

```bash
# CPU version (recommended for most users)
uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

## Usage

### Process a Single Screenshot

```bash
uv run python debug/test_single_frame.py screenshots/1.png
```

This runs the full recognition pipeline and outputs:
- Cell crops to `output/test/crops/`
- Preprocessed images to `output/test/preprocessed/`
- Extracted data to `output/test/output.csv`

Example output:
```
============================================================
EXTRACTED DATA
============================================================

Team 1 (Cyan)
----------------------------------------
  [   tank] reinhardt       PlayerName
           Ult: READY
           E:4 A:1 D:1 DMG:5214 HEAL:0 MIT:3784

  [    dps] tracer          AnotherPlayer
           Ult: 67%
           E:8 A:3 D:2 DMG:12450 HEAL:0 MIT:0
  ...
```

### Programmatic Usage

```python
from src.recognize import process_frame
from src.utils import load_image

# Load and process a screenshot
image = load_image("screenshots/1.png")
players = process_frame(image)

# Access extracted data
for player in players:
    print(f"Team {player.team}: {player.name}")
    print(f"  Role: {player.role}, Hero: {player.hero}")
    print(f"  Ult: {'Ready' if player.ult_ready else f'{player.ult_charge}%'}")
    print(f"  Stats: E:{player.elims} A:{player.assists} D:{player.deaths}")
    print(f"         DMG:{player.damage} HEAL:{player.healing} MIT:{player.mit}")
```

### Batch Processing

```python
from pathlib import Path
from src.process import process_screenshots

screenshots = list(Path("screenshots").glob("*.png"))
dataframe = process_screenshots(screenshots, output_path=Path("output/results.csv"))
```

### Debug Tools

```bash
# Full pipeline test with all debug output
uv run python debug/test_single_frame.py

# Visualize detected scoreboard edges
uv run python debug/show_edges.py

# Visualize column boundaries
uv run python debug/show_columns.py

# Test role template matching
uv run python debug/test_role_matching.py

# Test hero template matching
uv run python debug/test_hero_matching.py

# Test ultimate status detection
uv run python debug/test_ultimate_detection.py

# Test digit recognition on stat columns
uv run python debug/test_digit_matching.py

# Test OCR on name and stat columns
uv run python debug/test_ocr.py

# Visualize preprocessing pipeline steps
uv run python debug/test_preprocessing.py

# Interactive coordinate picker for calibration
uv run python debug/region_picker.py
```

## Project Structure

```
scoreboard_parser/
├── src/
│   ├── recognize.py       # Core recognition (roles, heroes, ult, OCR)
│   ├── digit_matcher.py   # Digit template matching for stats
│   ├── utils.py           # Edge detection, cropping, file I/O
│   ├── regions.py         # Layout constants and column widths
│   ├── process.py         # DataFrame assembly and CSV/Parquet export
│   ├── crop_portraits.py  # Utility for extracting hero templates
│   └── capture.py         # Screenshot capture (stub for future)
├── debug/
│   ├── test_utilities.py        # Shared test infrastructure
│   ├── test_single_frame.py     # Full pipeline integration test
│   ├── test_digit_matching.py   # Digit template matching test
│   ├── test_role_matching.py    # Role recognition test
│   ├── test_hero_matching.py    # Hero recognition test
│   ├── test_ultimate_detection.py  # Ult status detection test
│   ├── test_ocr.py              # PaddleOCR test
│   ├── test_preprocessing.py    # Preprocessing visualization
│   ├── show_edges.py            # Edge detection visualization
│   ├── show_columns.py          # Column boundary visualization
│   └── region_picker.py         # Interactive coordinate picker
├── templates/
│   ├── digits/            # Digit templates (0-9) for stat recognition
│   ├── heroes_cropped/    # Hero portrait templates
│   ├── roles/             # Role icon templates (tank, dps, support)
│   └── ult/               # Ultimate ready checkmark templates
├── screenshots/           # Input screenshots
└── output/                # Generated output files
```

## Technical Details

### Edge Detection

The scoreboard width varies based on hero perk configurations. The parser scans horizontally at the expected scoreboard height and detects sharp brightness gradients to find the left and right edges dynamically.

### Column Layout

Column positions are calculated from detected edges using known column widths:
- Left section: role icon (31px), hero portrait (61px), ult status (54px)
- Center section: player name (170px), perks (variable), report button (40px)
- Right section: elims/assists/deaths (55px each), damage/healing/mit (103px each)

### Template Matching

Role and ultimate icons use saturation-based extraction. White icons have low saturation regardless of the background color, allowing a single set of templates to work for both teams.

Hero portraits are matched directly against pre-cropped templates with separate variants for each team's background color.

### Digit Recognition

Statistics use a custom digit template matcher with:
1. 4x upscaling for improved resolution
2. CLAHE contrast enhancement
3. Otsu thresholding for binarization
4. Sliding window template matching
5. Non-maximum suppression to filter overlapping detections

### OCR Pipeline

Player names and ultimate percentages use PaddleOCR with:
1. 3x upscaling
2. HSV saturation thresholding to isolate white text
3. Binary conversion for cleaner input

## Data Output

The `PlayerData` class contains all extracted fields:

| Field | Type | Description |
|-------|------|-------------|
| team | int | Team number (1 = cyan, 2 = yellow) |
| row | int | Row index within team (0-4) |
| role | str | Player role (tank, dps, support) |
| hero | str | Hero name |
| name | str | Player name |
| ult_ready | bool | Whether ultimate is ready |
| ult_charge | int | Ultimate charge percentage (0-100) |
| elims | int | Eliminations |
| assists | int | Assists |
| deaths | int | Deaths |
| damage | int | Damage dealt |
| healing | int | Healing done |
| mit | int | Damage mitigated |

## Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Image processing and template matching |
| paddleocr | Text recognition engine |
| paddlepaddle | PaddleOCR backend |
| pandas | DataFrame operations and export |
| pillow | Image loading |
| mss | Screen capture |
| imagehash | Duplicate frame detection |
| matplotlib | Visualization (debug tools) |

## Known Limitations

- Requires 1920x1080 resolution screenshots (hardcoded region coordinates)
- OCR may occasionally misread thin italic characters
- Hero recognition requires the hero to be alive (greyed-out portraits return no match)
- Large numbers with visually similar digits may occasionally be misread

## Planned Features

- Automated screenshot capture during replay playback
- Event detection from stat deltas (kills, deaths, ult usage, hero swaps)
- Support for additional screen resolutions

## License

MIT
