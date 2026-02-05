"""Digit template matcher for stat recognition."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import TEMPLATES_DIR


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
        """Load and preprocess digit templates (0-9)."""
        templates = {}
        digits_dir = self._templates_dir / "digits"

        for digit in range(10):
            path = digits_dir / f"{digit}.png"
            if path.exists():
                image = cv2.imread(str(path))
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        processed = _preprocess(crop)
        raw_matches = _match_digits(processed, self.templates)
        filtered_matches = _non_max_suppression(raw_matches)

        if not filtered_matches:
            return None

        number_str = _matches_to_number(filtered_matches)

        if not number_str:
            return None

        try:
            return int(number_str)
        except ValueError:
            return None


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Preprocess crop: 4x scale -> grayscale -> CLAHE -> Otsu threshold."""
    scaled = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _match_digits(
    image: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.7,
) -> list[DigitMatch]:
    """Find all digit matches using sliding window template matching."""
    matches = []

    for digit, template in templates.items():
        template_height, template_width = template.shape[:2]
        image_height, image_width = image.shape[:2]

        if template_height > image_height or template_width > image_width:
            continue

        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for y, x in zip(*locations):
            confidence = float(result[y, x])
            matches.append(
                DigitMatch(
                    digit=digit,
                    x=int(x),
                    y=int(y),
                    confidence=confidence,
                    width=template_width,
                    height=template_height,
                )
            )

    return matches


def _non_max_suppression(
    matches: list[DigitMatch],
    min_distance: int = 20,
) -> list[DigitMatch]:
    """Remove overlapping matches, keeping highest confidence."""
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda m: -m.confidence)

    kept = []
    for match in sorted_matches:
        overlaps = False
        for kept_match in kept:
            if abs(match.x - kept_match.x) < min_distance:
                overlaps = True
                break

        if not overlaps:
            kept.append(match)

    return kept


def _matches_to_number(matches: list[DigitMatch]) -> str:
    """Convert matches to number string, sorted left-to-right."""
    sorted_matches = sorted(matches, key=lambda m: m.x)
    return "".join(m.digit for m in sorted_matches)


_digit_matcher: DigitMatcher | None = None


def get_digit_matcher() -> DigitMatcher:
    global _digit_matcher
    if _digit_matcher is None:
        _digit_matcher = DigitMatcher(TEMPLATES_DIR)
    return _digit_matcher


def recognize_stat(crop: np.ndarray) -> int | None:
    return get_digit_matcher().recognize(crop)
