"""Interactive tool to pick coordinates from a screenshot.

Usage:
    uv run python src/region_picker.py screenshots/your_image.png

Controls:
    Left click:  Mark a point (prints coordinates)
    Right click: Clear all points
    R key:       Toggle rectangle mode (click two corners to define a box)
    Q/Escape:    Quit
"""

import sys
from pathlib import Path

import cv2

# State
points: list[tuple[int, int]] = []
rect_mode = False


def mouse_callback(event: int, x: int, y: int, flags: int, param: tuple) -> None:
    """Handle mouse events."""
    global points
    img, original = param

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point: ({x}, {y})")

        if rect_mode and len(points) >= 2:
            p1, p2 = points[-2], points[-1]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            w, h = x2 - x1, y2 - y1
            print(f"  Rectangle: x={x1}, y={y1}, w={w}, h={h}")
            print(f"  As tuple:  ({x1}, {y1}, {x2}, {y2})")

        redraw(img, original)

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()
        print("Cleared all points")
        redraw(img, original)


def redraw(img, original) -> None:
    """Redraw the image with all marked points."""
    img[:] = original.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw rectangles between consecutive point pairs in rect mode
    if rect_mode:
        for i in range(0, len(points) - 1, 2):
            p1, p2 = points[i], points[i + 1]
            cv2.rectangle(img, p1, p2, (0, 255, 255), 2)

    cv2.imshow("Region Picker", img)


def main() -> None:
    global rect_mode

    if len(sys.argv) < 2:
        # Try to find first image in screenshots folder
        screenshots = Path("screenshots")
        if screenshots.exists():
            images = list(screenshots.glob("*.png")) + list(screenshots.glob("*.jpg"))
            if images:
                image_path = images[0]
                print(f"No image specified, using: {image_path}")
            else:
                print("Usage: uv run python src/region_picker.py <image_path>")
                print("No images found in screenshots/")
                sys.exit(1)
        else:
            print("Usage: uv run python src/region_picker.py <image_path>")
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

    img = original.copy()
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    print()
    print("Controls:")
    print("  Left click:  Mark point")
    print("  Right click: Clear all")
    print("  R:           Toggle rectangle mode")
    print("  Q/Escape:    Quit")
    print()

    cv2.namedWindow("Region Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Region Picker", min(w, 1600), min(h, 900))
    cv2.setMouseCallback("Region Picker", mouse_callback, (img, original))
    cv2.imshow("Region Picker", img)

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
