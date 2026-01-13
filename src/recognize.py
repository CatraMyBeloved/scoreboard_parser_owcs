"""Recognition module for scoreboard data extraction.

Handles:
- Role icon template matching
- Hero portrait template matching (TODO)
- Ult status detection (TODO)
- OCR for player names and stats (TODO)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np

from .utils import TEMPLATES_DIR

# Type aliases
Image: TypeAlias = np.ndarray
Templates: TypeAlias = dict[str, Image]


# -----------------------------------------------------------------------------
# Template Management
# -----------------------------------------------------------------------------


@dataclass
class TemplateMatch:
    """Result of a template matching operation."""

    name: str
    confidence: float

    def __bool__(self) -> bool:
        """Return True if this is a valid match (confidence > threshold)."""
        return self.confidence > 0.5


class TemplateManager:
    """Manages loading and caching of template images."""

    def __init__(self, templates_dir: Path = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self._cache: dict[str, Templates] = {}

    def load_templates(self, category: str) -> Templates:
        """Load all templates from a category folder.

        Args:
            category: Subfolder name (e.g., "roles", "heroes")

        Returns:
            Dict mapping template name to BGR image
        """
        if category in self._cache:
            return self._cache[category]

        templates = {}
        category_dir = self.templates_dir / category

        if not category_dir.exists():
            return templates

        for path in category_dir.glob("*.png"):
            img = cv2.imread(str(path))
            if img is not None:
                templates[path.stem] = img

        self._cache[category] = templates
        return templates

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()


# Global template manager instance
_template_manager = TemplateManager()


def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    return _template_manager


# -----------------------------------------------------------------------------
# Image Processing Utilities
# -----------------------------------------------------------------------------


def extract_by_saturation(image: Image, threshold: int = 50) -> Image:
    """Extract low-saturation regions (white) from an image.

    White has low saturation regardless of background color.
    This works for extracting white icons from colored backgrounds.

    Args:
        image: BGR image
        threshold: Saturation threshold (pixels below this are considered white)

    Returns:
        Binary image where white regions are 255, colored regions are 0
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary = cv2.threshold(saturation, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


# -----------------------------------------------------------------------------
# Role Recognition
# -----------------------------------------------------------------------------


def match_role(crop: Image, templates: Templates | None = None) -> TemplateMatch:
    """Match a role icon crop against templates.

    Uses saturation-based extraction to isolate the white icon shape,
    making it work for both blue (team 1) and yellow (team 2) backgrounds.

    Args:
        crop: BGR image of the role column cell
        templates: Optional pre-loaded templates (uses cached if not provided)

    Returns:
        TemplateMatch with role name and confidence score
    """
    if templates is None:
        templates = _template_manager.load_templates("roles")

    if not templates:
        return TemplateMatch(name="unknown", confidence=0.0)

    crop_binary = extract_by_saturation(crop)

    best_match = TemplateMatch(name="unknown", confidence=-1.0)

    for name, template in templates.items():
        template_binary = extract_by_saturation(template)

        # Resize template to match crop dimensions
        if template_binary.shape != crop_binary.shape:
            template_binary = cv2.resize(
                template_binary, (crop_binary.shape[1], crop_binary.shape[0])
            )

        # Template matching on binary images
        result = cv2.matchTemplate(crop_binary, template_binary, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])

        if score > best_match.confidence:
            best_match = TemplateMatch(name=name, confidence=score)

    return best_match


# -----------------------------------------------------------------------------
# Hero Recognition (TODO)
# -----------------------------------------------------------------------------


def match_hero(crop: Image, templates: Templates | None = None) -> TemplateMatch:
    """Match a hero portrait crop against templates.

    Args:
        crop: BGR image of the hero column cell
        templates: Optional pre-loaded templates

    Returns:
        TemplateMatch with hero name and confidence score
    """
    # TODO: Implement hero template matching
    # May need different approach than roles (color-based matching, histogram comparison, etc.)
    return TemplateMatch(name="unknown", confidence=0.0)


# -----------------------------------------------------------------------------
# Ult Status Detection (TODO)
# -----------------------------------------------------------------------------


def detect_ult_status(crop: Image) -> tuple[bool, int | None]:
    """Detect ultimate status from the ult column cell.

    Args:
        crop: BGR image of the ult column cell

    Returns:
        Tuple of (is_ready, charge_percentage)
        - is_ready: True if ult is ready (checkmark visible)
        - charge_percentage: 0-99 if charging, None if ready
    """
    # TODO: Implement ult detection
    # Could use:
    # - Template matching for the checkmark
    # - OCR for the percentage number
    # - Color detection (ready vs charging may have different colors)
    return False, None


# -----------------------------------------------------------------------------
# OCR Functions
# -----------------------------------------------------------------------------

# Lazy-loaded OCR instance
_ocr_instance = None


def _get_ocr():
    """Get or initialize PaddleOCR instance (lazy loading)."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(lang='en')
    return _ocr_instance


def _preprocess_for_ocr(crop: Image, scale: int = 3) -> Image:
    """Preprocess crop for OCR using saturation extraction.

    Extracts white text from colored backgrounds (blue/yellow).
    White has low saturation, so we threshold on saturation channel.
    Upscales for better OCR accuracy on small crops.
    """
    # Upscale first for better quality
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Low saturation = white text, high saturation = colored background
    # Threshold so text becomes black on white (better for OCR)
    _, binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

    # Dilate to thicken thin characters (like italic "I")
    # We want to erode because text is black on white (erode black = thicken)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)

    # Convert back to BGR (3 channel) for PaddleOCR
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _run_ocr(crop: Image, preprocess: bool = True) -> str:
    """Run OCR on a crop and return raw text."""
    ocr = _get_ocr()

    # Apply preprocessing for better text extraction
    if preprocess:
        crop = _preprocess_for_ocr(crop)

    result = ocr.predict(crop)
    # Result is a list of dicts, first element contains 'rec_texts'
    if result and len(result) > 0:
        first = result[0]
        if isinstance(first, dict) and 'rec_texts' in first:
            texts = first['rec_texts']
            if texts:
                return ' '.join(texts)
    return ''


def ocr_text(crop: Image) -> str:
    """Extract text from a cropped region using OCR.

    Args:
        crop: BGR image containing text

    Returns:
        Extracted text string
    """
    return _run_ocr(crop)


def ocr_number(crop: Image) -> int | None:
    """Extract a number from a cropped region.

    Args:
        crop: BGR image containing a number

    Returns:
        Parsed integer or None if extraction failed
    """
    text = _run_ocr(crop)
    if not text:
        return None
    # Remove commas and whitespace
    cleaned = text.replace(',', '').replace(' ', '').strip()
    # Try to parse as integer
    try:
        return int(cleaned)
    except ValueError:
        return None


# -----------------------------------------------------------------------------
# Frame Processing
# -----------------------------------------------------------------------------


@dataclass
class PlayerData:
    """Data extracted for a single player."""

    team: int
    row: int
    role: str | None = None
    hero: str | None = None
    name: str | None = None
    ult_ready: bool | None = None
    ult_charge: int | None = None
    elims: int | None = None
    assists: int | None = None
    deaths: int | None = None
    damage: int | None = None
    healing: int | None = None
    mit: int | None = None


def process_player(
    image: Image,
    team: int,
    row: int,
    columns: dict[str, tuple[int, int]],
) -> PlayerData:
    """Extract all data for a single player.

    Args:
        image: Full screenshot
        team: Team number (1 or 2)
        row: Row index (0-4)
        columns: Pre-calculated column positions

    Returns:
        PlayerData with all extracted fields
    """
    from .utils import crop_cell
    from . import regions

    data = PlayerData(team=team, row=row)

    # Role recognition
    role_crop = crop_cell(image, team, row, "role", columns)
    role_match = match_role(role_crop)
    if role_match:
        data.role = role_match.name

    # Hero recognition (TODO)
    # hero_crop = crop_cell(image, team, row, "hero", columns)
    # hero_match = match_hero(hero_crop)
    # if hero_match:
    #     data.hero = hero_match.name

    # Ult status (TODO)
    # ult_crop = crop_cell(image, team, row, "ult", columns)
    # data.ult_ready, data.ult_charge = detect_ult_status(ult_crop)

    # Name OCR
    name_crop = crop_cell(image, team, row, "name", columns)
    data.name = ocr_text(name_crop)

    # Stats OCR
    for stat in ["elims", "assists", "deaths", "damage", "healing", "mit"]:
        stat_crop = crop_cell(image, team, row, stat, columns)
        setattr(data, stat, ocr_number(stat_crop))

    return data


def process_frame(image: Image) -> list[PlayerData]:
    """Process a screenshot and extract all player data.

    Args:
        image: Full screenshot as BGR numpy array

    Returns:
        List of PlayerData for all 10 players (5 per team)
    """
    from .utils import detect_scoreboard_edges, calculate_column_positions

    # Detect scoreboard and calculate column positions
    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    # Process all players
    players = []
    for team in (1, 2):
        for row in range(5):
            player = process_player(image, team, row, columns)
            players.append(player)

    return players
