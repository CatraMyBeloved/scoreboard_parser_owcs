"""Screenshot capture loop for Overwatch 2 replay viewer."""

from .regions import CAPTURE_INTERVAL


def capture_loop() -> None:
    """Main capture loop."""
    # TODO: Implement capture loop
    pass


def detect_duplicate(frame1_path: str, frame2_path: str) -> bool:
    """Check if two frames are duplicates using image hashing."""
    # TODO: Implement using imagehash
    pass


if __name__ == "__main__":
    capture_loop()
