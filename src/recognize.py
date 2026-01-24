"""Recognition module for scoreboard data extraction.

Handles:
- Role icon template matching
- Hero portrait template matching
- Ult status detection
- OCR for player names (PaddleOCR)
- Digit template matching for stats
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeAlias

import cv2
import numpy as np

from .utils import TEMPLATES_DIR

# Type aliases
Image: TypeAlias = np.ndarray
Templates: TypeAlias = dict[str, Image]
Preprocessor: TypeAlias = Callable[[Image], Image]


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
            image = cv2.imread(str(path))
            if image is not None:
                templates[path.stem] = image

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


def find_best_template_match(
    crop: Image,
    templates: Templates,
    preprocess: Preprocessor | None = None,
    name_transform: Callable[[str], str] | None = None,
) -> TemplateMatch:
    """Find the best matching template for a crop.

    Common helper for template matching operations. Handles resizing templates
    to match crop dimensions and tracks the best match.

    Args:
        crop: BGR image to match against
        templates: Dict mapping template names to BGR images
        preprocess: Optional function to preprocess both crop and templates
        name_transform: Optional function to transform template name to match name

    Returns:
        TemplateMatch with best match name and confidence
    """
    if preprocess is not None:
        crop_processed = preprocess(crop)
    else:
        crop_processed = crop

    best_match = TemplateMatch(name="unknown", confidence=-1.0)

    for template_name, template in templates.items():
        if preprocess is not None:
            template_processed = preprocess(template)
        else:
            template_processed = template

        if template_processed.shape != crop_processed.shape:
            template_processed = cv2.resize(
                template_processed, (crop_processed.shape[1], crop_processed.shape[0])
            )

        score = float(cv2.matchTemplate(crop_processed, template_processed, cv2.TM_CCOEFF_NORMED)[0, 0])

        if score > best_match.confidence:
            match_name = name_transform(template_name) if name_transform else template_name
            best_match = TemplateMatch(name=match_name, confidence=score)

    return best_match


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

    return find_best_template_match(crop, templates, preprocess=extract_by_saturation)


# -----------------------------------------------------------------------------
# Hero Recognition
# -----------------------------------------------------------------------------


HERO_CONFIDENCE_THRESHOLD = 0.7


def _hero_name_from_template(template_name: str) -> str:
    """Extract hero name from template filename (removes _1/_2 suffix)."""
    return template_name.rsplit("_", 1)[0]


def match_hero(crop: Image, templates: Templates | None = None) -> TemplateMatch | None:
    """Match a hero portrait crop against templates.

    Uses direct template matching against pre-cropped hero portraits.
    Templates include both cyan (team 1) and yellow (team 2) background variants.

    Args:
        crop: BGR image of the hero column cell
        templates: Optional pre-loaded templates (uses cached if not provided)

    Returns:
        TemplateMatch with hero name and confidence score, or None if no confident match
        (e.g., hero is dead/greyed out).
    """
    if templates is None:
        templates = _template_manager.load_templates("heroes_cropped")

    if not templates:
        return None

    best_match = find_best_template_match(
        crop, templates, name_transform=_hero_name_from_template
    )

    if best_match.confidence < HERO_CONFIDENCE_THRESHOLD:
        return None

    return best_match


# -----------------------------------------------------------------------------
# Ult Status Detection
# -----------------------------------------------------------------------------


ULT_READY_CONFIDENCE_THRESHOLD = 0.7


def detect_ultimate_ready_checkmark(crop: Image, templates: Templates) -> bool:
    """Detect if the ultimate ready checkmark is visible.

    Args:
        crop: BGR image of the ult column cell
        templates: Pre-loaded ult templates (checkmark images)

    Returns:
        True if checkmark is detected with sufficient confidence
    """
    if not templates:
        return False

    crop_binary = extract_by_saturation(crop)
    best_score = -1.0

    for template in templates.values():
        template_binary = extract_by_saturation(template)

        if template_binary.shape != crop_binary.shape:
            template_binary = cv2.resize(
                template_binary, (crop_binary.shape[1], crop_binary.shape[0])
            )

        best_score = max(best_score, float(cv2.matchTemplate(crop_binary, template_binary, cv2.TM_CCOEFF_NORMED)[0, 0]))

    return best_score >= ULT_READY_CONFIDENCE_THRESHOLD


def read_ultimate_charge_percentage(crop: Image) -> int | None:
    """Read the ultimate charge percentage via OCR.

    Args:
        crop: BGR image of the ult column cell

    Returns:
        Charge percentage (0-100) or None if unreadable
    """
    text = _run_ocr(crop)
    digits = "".join(c for c in text if c.isdigit())

    if not digits:
        return None

    try:
        percentage = int(digits)
        return percentage if 0 <= percentage <= 100 else None
    except ValueError:
        return None


def detect_ult_status(crop: Image, templates: Templates | None = None) -> tuple[bool, int | None]:
    """Detect ultimate status from the ult column cell.

    First tries to match the checkmark (ult ready). If no match, uses OCR
    to read the charge percentage.

    Args:
        crop: BGR image of the ult column cell
        templates: Optional pre-loaded ult templates

    Returns:
        Tuple of (is_ready, charge_percentage)
        - is_ready: True if ult is ready (checkmark visible)
        - charge_percentage: 0-99 if charging, None if ready
    """
    if templates is None:
        templates = _template_manager.load_templates("ult")

    if detect_ultimate_ready_checkmark(crop, templates):
        return True, None

    percentage = read_ultimate_charge_percentage(crop)
    if percentage is not None:
        return False, percentage

    return False, None


# -----------------------------------------------------------------------------
# OCR Functions (PaddleOCR 3.x)
# -----------------------------------------------------------------------------

_paddle_ocr = None


def _get_paddle_ocr():
    """Get or initialize PaddleOCR instance (lazy loading)."""
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(lang="en")
    return _paddle_ocr


def _preprocess_for_ocr(crop: Image, scale: int = 3) -> Image:
    """Preprocess crop for OCR using saturation extraction.

    Extracts white text from colored backgrounds (blue/yellow).
    White has low saturation, so we threshold on saturation channel.

    Args:
        crop: BGR image
        scale: Upscale factor

    Returns:
        BGR image ready for OCR
    """
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _extract_text_from_ocr_result(result) -> str:
    """Extract text from PaddleOCR result structure.

    PaddleOCR returns a list where the first element is a dict with 'rec_texts'.

    Args:
        result: Raw PaddleOCR result

    Returns:
        Extracted text or empty string
    """
    if not result or len(result) == 0:
        return ""

    first = result[0]
    if not isinstance(first, dict) or "rec_texts" not in first:
        return ""

    texts = first["rec_texts"]
    return " ".join(texts) if texts else ""


def _run_ocr(crop: Image) -> str:
    """Run OCR on a crop and return raw text."""
    return _extract_text_from_ocr_result(_get_paddle_ocr().predict(_preprocess_for_ocr(crop)))


def ocr_text(crop: Image) -> str:
    """Extract text from a cropped region using PaddleOCR.

    Args:
        crop: BGR image containing text

    Returns:
        Extracted text string
    """
    return _run_ocr(crop)


def ocr_ult_charge(crop: Image) -> int | None:
    """Extract ult charge percentage from a cropped region.

    Uses PaddleOCR to read the percentage number (digits only).

    Args:
        crop: BGR image containing the ult charge percentage

    Returns:
        Parsed integer (0-100) or None if extraction failed
    """
    return read_ultimate_charge_percentage(crop)


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
    from .digit_matcher import recognize_stat

    data = PlayerData(team=team, row=row)

    role_crop = crop_cell(image, team, row, "role", columns)
    role_match = match_role(role_crop)
    if role_match:
        data.role = role_match.name

    hero_crop = crop_cell(image, team, row, "hero", columns)
    hero_match = match_hero(hero_crop)
    if hero_match:
        data.hero = hero_match.name

    ult_crop = crop_cell(image, team, row, "ult", columns)
    data.ult_ready, data.ult_charge = detect_ult_status(ult_crop)

    name_crop = crop_cell(image, team, row, "name", columns)
    data.name = ocr_text(name_crop)

    for stat in ["elims", "assists", "deaths", "damage", "healing", "mit"]:
        stat_crop = crop_cell(image, team, row, stat, columns)
        setattr(data, stat, recognize_stat(stat_crop))

    return data


def process_frame(image: Image) -> list[PlayerData]:
    """Process a screenshot and extract all player data.

    Args:
        image: Full screenshot as BGR numpy array

    Returns:
        List of PlayerData for all 10 players (5 per team)
    """
    from .utils import detect_scoreboard_edges, calculate_column_positions

    left_edge, right_edge = detect_scoreboard_edges(image)
    columns = calculate_column_positions(left_edge, right_edge)

    players = []
    for team in (1, 2):
        for row in range(5):
            player = process_player(image, team, row, columns)
            players.append(player)

    return players
