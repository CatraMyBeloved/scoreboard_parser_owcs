# Overwatch Replay Stats Extractor

## Project Goal

Extract player statistics from Overwatch 2 replay viewer at regular intervals to build a time series dataset. This enables analysis of stat progression, ult economy, hero switches, and other match dynamics.

## Output Data

For each player, at each timestamp:

| Field | Type | Source |
|-------|------|--------|
| timestamp | float | Derived from frame number × interval |
| team | string | Row position (top 5 vs bottom 5) |
| player_name | string | OCR (cached) |
| hero | string | Template matching |
| role | string | Template matching (DPS/Tank/Support) |
| ult_charge | int (0-100) or "ready" | OCR or template match |
| eliminations | int | OCR |
| assists | int | OCR |
| deaths | int | OCR |
| damage | int | OCR |
| healing | int | OCR |
| damage_mitigated | int | OCR |

## Detectable Events (Post-Processing)

- Hero switches (portrait changes)
- Deaths (death count increments)
- Kills (elim count increments)
- Ult gained (charge hits 100% / ready)
- Ult used (ready → 0%)

---

## Technical Architecture

### Phase 1: Capture

User positions replay at 00:00:00 and runs script. Script takes control:

```
Loop:
  1. Press Tab (show scoreboard)
  2. Wait ~100ms for render
  3. Screenshot
  4. Release Tab
  5. Wait for interval
  6. Check for duplicate frame → exit if replay ended
```

**Libraries**: `mss` (screenshots), `pynput` (keyboard), `imagehash` (duplicate detection)

### Phase 2: Recognition

Process saved screenshots:

```
For each frame:
  1. Crop into individual stat regions
  2. Template match: hero portraits, role icons, ult ready symbol
  3. OCR: player names (first frame only), all stat numbers, ult percentage
  4. Parse and validate values
```

**Libraries**: `opencv-python` (template matching, cropping), `paddleocr` (text recognition), `pillow` (image handling)

### Phase 3: Processing

Transform raw extractions into clean dataset:

```
1. Assemble all frames into single DataFrame
2. Clean/validate values (strip commas, handle OCR errors)
3. Derive events from deltas (kills, deaths, ult usage, hero swaps)
4. Export to CSV / Parquet
```

**Libraries**: `pandas`

---

## Project Structure

```
overwatch-stats/
├── pyproject.toml
├── uv.lock
├── PROJECT_PLAN.md
│
├── src/
│   ├── capture.py           # Screenshot loop
│   ├── recognize.py         # OCR + template matching
│   ├── process.py           # Build time series, detect events
│   ├── regions.py           # Bounding box definitions
│   └── utils.py             # Shared helpers
│
├── templates/
│   ├── heroes/              # Hero portrait PNGs
│   │   ├── ana.png
│   │   ├── winston.png
│   │   └── ...
│   ├── roles/               # Role icon PNGs
│   │   ├── dps.png
│   │   ├── tank.png
│   │   └── support.png
│   └── ult_ready.png        # Checkmark symbol
│
├── screenshots/             # Raw captures (gitignored)
│
└── output/                  # Final CSVs (gitignored)
```

---

## Configuration

All resolution-dependent values in one place (`regions.py`):

```python
RESOLUTION = (1920, 1080)
CAPTURE_INTERVAL = 5.0  # seconds

# Row Y positions (top of each player row)
TEAM1_ROWS = [y1, y2, y3, y4, y5]
TEAM2_ROWS = [y6, y7, y8, y9, y10]

# Column X ranges (start, end)
COLUMNS = {
    "ult":    (x1, x2),
    "role":   (x3, x4),
    "hero":   (x5, x6),
    "name":   (x7, x8),
    "elims":  (x9, x10),
    "assists": (x11, x12),
    "deaths": (x13, x14),
    "damage": (x15, x16),
    "healing": (x17, x18),
    "mit":    (x19, x20),
}

ROW_HEIGHT = ??  # pixels
```

**TODO**: Measure actual pixel values from reference screenshot.

---

## Dependencies

```
mss           # Fast screenshots
pynput        # Keyboard control
opencv-python # Image processing, template matching
pillow        # Image handling
paddleocr     # Text recognition
pandas        # Data wrangling
imagehash     # Duplicate frame detection
```

---

## Performance Estimates

### Capture Phase
- Runtime: Same as replay length
- Resource usage: Negligible

### Recognition Phase (240 frames, 20 min replay @ 5s interval)

| Setup | Time |
|-------|------|
| PaddleOCR, CPU | 20-60 min |
| PaddleOCR, GPU | 5-10 min |

---

## Implementation Order

### Step 1: Setup
- [x] Define project plan
- [x] Initialize uv project
- [x] Install dependencies
- [x] Create folder structure

### Step 2: Region Mapping
- [ ] Take reference screenshot
- [ ] Measure all bounding boxes
- [ ] Define in `regions.py`
- [ ] Write crop helper functions

### Step 3: Template Library
- [ ] Extract all hero portraits
- [ ] Extract role icons
- [ ] Extract ult ready symbol
- [ ] Test template matching accuracy

### Step 4: Recognition Module
- [ ] Integrate PaddleOCR
- [ ] Test on sample stat crops
- [ ] Tune preprocessing if needed
- [ ] Batch OCR calls per frame
- [ ] Handle edge cases (commas, empty values)

### Step 5: Processing Module
- [ ] Assemble raw data into DataFrame
- [ ] Validate/clean values
- [ ] Derive events from deltas
- [ ] Export functionality

### Step 6: Capture Module (Harness)
- [ ] Basic screenshot loop
- [ ] Keyboard control (Tab hold)
- [ ] Duplicate detection for auto-stop
- [ ] Test with real replay

### Step 7: Integration
- [ ] End-to-end test on full replay
- [ ] Error handling
- [ ] CLI interface

---

## Known Challenges

| Issue | Mitigation |
|-------|------------|
| OCR misreads stylized font | Preprocessing (threshold, contrast), or fine-tune model |
| Player names with unicode | PaddleOCR handles most; may need manual correction |
| Ult percentage in circle | May need custom crop shape or preprocessing |
| Dead player row styling | Check if grayed out affects OCR; handle separately |
| Mid-match hero swap | Re-OCR name when hero changes |

---

## Future Enhancements (Out of Scope for v1)

- Kill feed parsing (who killed whom)
- Team fight detection
- Map/mode detection from loading screen
- Multi-resolution support
- GUI for region calibration