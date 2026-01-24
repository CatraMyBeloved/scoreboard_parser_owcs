"""Interactive tool to pick coordinates from a screenshot.

Usage:
    uv run python debug/region_picker.py screenshots/your_image.png

Controls:
    Left click:  Mark a point (prints coordinates)
    Right click: Clear all points
    R key:       Toggle rectangle mode (click two corners to define a box)
    Q/Escape:    Quit
"""

import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import SCREENSHOTS_DIR

# State
points: list[tuple[int, int]] = []
rect_mode = False


def mouse_callback(event: int, x: int, y: int, flags: int, param: tuple) -> None:
    """Handle mouse events."""
    global points
    image, original = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point: ({x}, {y})")

        if rect_mode and len(points) >= 2:
            p1, p2 = points[-2], points[-1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            width, height = x2 - x1, y2 - y1
            print(f"  Rectangle: x={x1}, y={y1}, w={width}, h={height}")
            print(f"  As tuple:  ({x1}, {y1}, {x2}, {y2})")

        redraw(image, original)

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()
        print("Cleared all points")
        redraw(image, original)


def redraw(image, original) -> None:
    """Redraw the image with all marked points."""
    image[:] = original.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw rectangles between consecutive point pairs in rect mode
    if rect_mode:
        for i in range(0, len(points) - 1, 2):
            p1, p2 = points[i], points[i + 1]
            cv2.rectangle(image, p1, p2, (0, 255, 255), 2)

    cv2.imshow("Region Picker", image)


def main() -> None:
    global rect_mode

    if len(sys.argv) < 2:
        # Try to find first image in screenshots folder
        if SCREENSHOTS_DIR.exists():
            images = list(SCREENSHOTS_DIR.glob("*.png")) + list(SCREENSHOTS_DIR.glob("*.jpg"))
            if images:
                image_path = images[0]
                print(f"No image specified, using: {image_path}")
            else:
                print("Usage: uv run python debug/region_picker.py <image_path>")
                print("No images found in screenshots/")
                sys.exit(1)
        else:
            print("Usage: uv run python debug/region_picker.py <image_path>")
            sys.exit(1)
    else:
        image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    image = original.copy()
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")
    print()
    print("Controls:")
    print("  Left click:  Mark point")
    print("  Right click: Clear all")
    print("  R:           Toggle rectangle mode")
    print("  Q/Escape:    Quit")
    print()

    cv2.namedWindow("Region Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Region Picker", min(width, 1600), min(height, 900))
    cv2.setMouseCallback("Region Picker", mouse_callback, (image, original))
    cv2.imshow("Region Picker", image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or Escape
            break
        elif key == ord('r'):
            rect_mode = not rect_mode
            print(f"Rectangle mode: {'ON' if rect_mode else 'OFF'}")

    cv2.destroyAllWindows()

    # Print summary
    if points:
        print()
        print("=== Summary ===")
        print(f"All points: {points}")


if __name__ == "__main__":
    main()
