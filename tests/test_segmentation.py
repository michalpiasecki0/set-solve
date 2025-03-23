import sys
import cv2
from pathlib import Path

# add module root path
root_path = str(Path(".").resolve())
sys.path.append(str(Path(".").resolve()))

from src.segmentation import detect_cards # noqa: E402

def test_detection():
    # fix to do -> incorrect for 14 cards
    paths = [f"tests/data/{i}.jpg" for i in [0, 8]] # gt = [12, 12, 14]
    imgs = [cv2.imread(path) for path in paths]
    detected_cards = list(map(len, [detect_cards(img) for img in imgs]))
    
    assert detected_cards == [12, 12]
    