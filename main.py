"""
Main script.
Currently it only takes single img and saves all cards segmented from an image..
"""

import argparse

import cv2
import os

from src.solve import SetSolver
from src.net import MultiOutputCNN
from src.utils import get_model

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
    
    model_fn = MultiOutputCNN
    model_path = "/home/michal/personal/programming/set-solve/data/set_classifier_simple.pth"
    model = get_model(model_fn, model_path)
    
    
    
    solver = SetSolver(img, model, args.output_path)
    solver.predict_card_values()
    print("--------------------------------------------")
    print(solver.find_sets())
    solver.present_sets()