"""
Main script.
Currently it only takes single img and saves all cards segmented from an image.
"""

import argparse
import os

import cv2
import sys
from pathlib import Path
from typing import List

from src.segmentation import detect_cards
from src.card import DetectedCard
from src.utils import save_detected_cards




def parse_args():
    parser = argparse.ArgumentParser(description="Run set detection on img.")
    parser.add_argument("--img-path", type=str, required=True, help="Path to img")
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to the output folder"
    )
    parser.add_argument("--grayscale", action="store_true", help="Save grayscale")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = cv2.imread(args.img_path)
    detected_cards = detect_cards(img)
    save_detected_cards(detected_cards, args.output_path, args.grayscale)
