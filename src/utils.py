import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import List, Callable
from src.solve import DetectedCard


def cv2_imshow(img: np.ndarray):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


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
        cv2.imwrite(f"{output_dir}/{idx + start_idx}.jpg", card_processed)


def get_model(model: Callable, weights_path: str):
    """
    Load model with weights from weights_path.

    Args:
        model (Callable): Model class
        weights_path (str): Path to weights

    Returns:
        model: Model with loaded weights
    """
    model_instance = model()
    model_instance.load_state_dict(torch.load(weights_path))
    return model_instance