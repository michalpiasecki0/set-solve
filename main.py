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


def save_detected_cards(
    detected_cards: List[DetectedCard], output_dir: str, grayscale_img: bool
):
    """
    Save cards segmented on img as separate .jpg files to output_dir.

    Args:
        detected_cards (List[DetectedCard])
        output_dir (str)
        grayscale_img (bool): if set, save grayscale imgs
    """
    os.makedirs(output_dir, exist_ok=True)
    start_idx = len(os.listdir(output_dir))
    for idx, card in enumerate(detected_cards):
        card_processed = (
            cv2.cvtColor(card.img_bgr, cv2.COLOR_BGR2GRAY)
            if grayscale_img
            else card.img_bgr
        )
        cv2.imwrite(f"{output_dir}/{idx+start_idx}.jpg", card_processed)


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
