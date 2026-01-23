"""Digit template matcher for stat recognition.

Uses sliding window template matching to recognize numbers from stat cells.
Templates are loaded lazily on first use.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import TEMPLATES_DIR


# -----------------------------------------------------------------------------
# Core Classes
# -----------------------------------------------------------------------------


@dataclass
class DigitMatch:
    """A detected digit match."""

    digit: str
    x: int
    y: int
    confidence: float
    width: int
    height: int


class DigitMatcher:
    """Digit template matcher with lazy-loaded templates."""

    def __init__(self, templates_dir: Path):
        self._templates: dict[str, np.ndarray] | None = None
        self._templates_dir = templates_dir

    def _load_templates(self) -> dict[str, np.ndarray]:
        """Load and preprocess digit templates (0-9).

        Templates are already 4x scaled, so we only convert to binary.

        Returns:
            Dict mapping digit string ('0'-'9') to binary template.
        """
        templates = {}
        digits_dir = self._templates_dir / "digits"

        for digit in range(10):
            path = digits_dir / f"{digit}.png"
            if path.exists():
                img = cv2.imread(str(path))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    templates[str(digit)] = binary

        return templates

    @property
    def templates(self) -> dict[str, np.ndarray]:
        """Get templates, loading them if needed."""
        if self._templates is None:
            self._templates = self._load_templates()
        return self._templates

    def recognize(self, crop: np.ndarray) -> int | None:
        """Recognize number from BGR crop.

        Args:
            crop: BGR image of a stat cell

        Returns:
            Recognized integer or None if recognition failed
        """
        if not self.templates:
            return None

        # Preprocess the crop
        processed = _preprocess(crop)

        # Find digit matches
        raw_matches = _match_digits(processed, self.templates)

        # Apply non-max suppression
        filtered_matches = _non_max_suppression(raw_matches)

        if not filtered_matches:
            return None

        # Convert to number string
        number_str = _matches_to_number(filtered_matches)

        if not number_str:
            return None

        try:
            return int(number_str)
        except ValueError:
            return None


# -----------------------------------------------------------------------------
# Private Functions
# -----------------------------------------------------------------------------


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Preprocess crop for template matching.

    Pipeline: 4x scale -> grayscale -> CLAHE -> Otsu threshold

    Args:
        crop: BGR image

    Returns:
        Binary image (white digits on black background)
    """
    # Scale 4x
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold (NOT inverted - keeps white digits on black background)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def _match_digits(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
) -> list[DigitMatch]:
    """Find all digit matches in an image using sliding window template matching.

    Args:
        image: Preprocessed binary image
        templates: Dict of digit -> binary template
        threshold: Minimum confidence for a match

    Returns:
        List of DigitMatch objects (unsorted, may have overlaps)
    """
    matches = []

    for digit, template in templates.items():
        th, tw = template.shape[:2]
        ih, iw = image.shape[:2]

        # Skip if template is larger than image
        if th > ih or tw > iw:
            continue

        # Template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Find all locations above threshold
        locations = np.where(result >= threshold)

        for y, x in zip(*locations):
            confidence = float(result[y, x])
            matches.append(
                DigitMatch(
                    digit=digit,
                    x=int(x),
                    y=int(y),
                    confidence=confidence,
                    width=tw,
                    height=th,
                )
            )

    return matches


def _non_max_suppression(
    matches: list[DigitMatch],
    min_distance: int = 20,
) -> list[DigitMatch]:
    """Remove overlapping matches, keeping highest confidence.

    Args:
        matches: List of digit matches
        min_distance: Minimum X distance between matches

    Returns:
        Filtered list with overlaps removed
    """
    if not matches:
        return []

    # Sort by confidence (highest first)
    sorted_matches = sorted(matches, key=lambda m: -m.confidence)

    kept = []
    for match in sorted_matches:
        # Check if this match overlaps with any kept match
        overlaps = False
        for kept_match in kept:
            if abs(match.x - kept_match.x) < min_distance:
                overlaps = True
                break

        if not overlaps:
            kept.append(match)

    return kept


def _matches_to_number(matches: list[DigitMatch]) -> str:
    """Convert sorted matches to a number string.

    Args:
        matches: List of DigitMatch objects

    Returns:
        String representation of the detected number
    """
    # Sort by X position (left to right)
    sorted_matches = sorted(matches, key=lambda m: m.x)
    return "".join(m.digit for m in sorted_matches)


# -----------------------------------------------------------------------------
# Module-level API
# -----------------------------------------------------------------------------


_digit_matcher: DigitMatcher | None = None


def get_digit_matcher() -> DigitMatcher:
    """Get or create global digit matcher instance."""
    global _digit_matcher
    if _digit_matcher is None:
        _digit_matcher = DigitMatcher(TEMPLATES_DIR)
    return _digit_matcher


def recognize_stat(crop: np.ndarray) -> int | None:
    """Convenience function for stat recognition.

    Args:
        crop: BGR image of a stat cell

    Returns:
        Recognized integer or None if recognition failed
    """
    return get_digit_matcher().recognize(crop)
