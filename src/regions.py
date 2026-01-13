"""Bounding box definitions for scoreboard regions.

All resolution-dependent values are defined here.
Measured at 1920x1080 resolution.
"""

RESOLUTION = (1920, 1080)
CAPTURE_INTERVAL = 5.0  # seconds

# Row Y positions (top of each player row)
# Calculated from team bounds: 5 players per team, evenly spaced
TEAM1_ROWS: list[int] = [206, 268, 330, 391, 453]  # 309px / 5 = ~62px each
TEAM2_ROWS: list[int] = [601, 663, 724, 786, 847]  # 308px / 5 = ~62px each
ROW_HEIGHT: int = 62  # pixels

# Column widths (fixed, anchored from edges)
# Left side columns (anchored from left edge)
LEFT_COLUMNS: dict[str, int] = {
    "role": 31,
    "hero": 61,
    "ult": 54,
}
LEFT_COLUMNS_ORDER = ["role", "hero", "ult"]
LEFT_TOTAL = sum(LEFT_COLUMNS.values())  # 146px

# Right side columns (anchored from right edge)
# Order: right-to-left for calculation, but stored left-to-right for display
RIGHT_COLUMNS: dict[str, int] = {
    "elims": 55,
    "assists": 55,
    "deaths": 55,
    "damage": 103,
    "healing": 103,
    "mit": 103,
}
RIGHT_COLUMNS_ORDER = ["elims", "assists", "deaths", "damage", "healing", "mit"]
RIGHT_TOTAL = sum(RIGHT_COLUMNS.values())  # 474px

# Report button on far right (skip this)
REPORT_BUTTON_WIDTH = 40

# Name section (fixed width)
NAME_WIDTH = 170

# Perks section (variable width, fills remaining gap between name and stats)
