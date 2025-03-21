# import torch
import argparse
import sys
from pathlib import Path

import cv2

# add module root path
root_path = str(Path(".").resolve())
sys.path.append(str(Path(".").resolve()))

from src.segmentation import detect_cards  # noqa: E402
from src.utils import save_detected_cards  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the data folder"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument("--grayscale", action="store_true", help="Save grayscale")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_paths = [str(p) for p in Path(args.path).rglob("*.jpg")]
    for img_path in img_paths:
        img = cv2.imread(img_path)
        detected_cards = detect_cards(img)
        save_detected_cards(detected_cards, args.output, args.grayscale)
