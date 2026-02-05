"""Bounding box definitions for scoreboard regions at 1920x1080."""

RESOLUTION = (1920, 1080)
CAPTURE_INTERVAL = 5.0

TEAM1_ROWS: list[int] = [206, 268, 330, 391, 453]
TEAM2_ROWS: list[int] = [601, 663, 724, 786, 847]
ROW_HEIGHT: int = 62

LEFT_COLUMNS: dict[str, int] = {
    "role": 31,
    "hero": 61,
    "ult": 54,
}
LEFT_COLUMNS_ORDER = ["role", "hero", "ult"]
LEFT_TOTAL = sum(LEFT_COLUMNS.values())

RIGHT_COLUMNS: dict[str, int] = {
    "elims": 55,
    "assists": 55,
    "deaths": 55,
    "damage": 103,
    "healing": 103,
    "mit": 103,
}
RIGHT_COLUMNS_ORDER = ["elims", "assists", "deaths", "damage", "healing", "mit"]
RIGHT_TOTAL = sum(RIGHT_COLUMNS.values())

REPORT_BUTTON_WIDTH = 40
NAME_WIDTH = 170
