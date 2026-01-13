"""Screenshot capture loop for Overwatch 2 replay viewer.

Handles:
- Tab key press to show scoreboard
- Screenshot capture at regular intervals
- Duplicate frame detection for auto-stop
"""

from .regions import CAPTURE_INTERVAL


def capture_loop() -> None:
    """Main capture loop.

    1. Press Tab (show scoreboard)
    2. Wait ~100ms for render
    3. Screenshot
    4. Release Tab
    5. Wait for interval
    6. Check for duplicate frame -> exit if replay ended
    """
    # TODO: Implement capture loop
    pass


def detect_duplicate(frame1_path: str, frame2_path: str) -> bool:
    """Check if two frames are duplicates using image hashing."""
    # TODO: Implement using imagehash
    pass


if __name__ == "__main__":
    capture_loop()
