import numpy as np
import os
from typing import List, Optional
from itertools import combinations

import torch
import torch.nn.functional as F
from src.constants import RUN_TRANSFORM, COUNT_MAPPING
from src.segmentation import detect_cards
import cv2


class DetectedCard:
    def __init__(
        self, img_bgr: np.ndarray, rect: np.ndarray, values: Optional[List[int]] = []
    ):
        """
        Initialize a card with an image, bounding rectangle and values.
        Value
        Args:
            img_bgr (np.ndarray): cropped bgr image from original img
            rect (np.ndarray): rectangle coordinates
            values (Optional[List[int]], optional): [number, color, shape, fill]
        """
        self.img_bgr = img_bgr
        self.rect = rect
        self._tensor_img = RUN_TRANSFORM(img_bgr)
        if values:
            assert len(values) == 4, (
                "Values must be a list of 4 integers, representing number, color, shape, fill"
            )

        self.values = values if values else []  # number, color, shape, fill
        self.values_human = []
    

    def __eq__(self, other):
        """
        Cards are equal when their values are equal.
        """
        return self.values == other.values

    def __hash__(self):
        """Make object hashable for comparison"""
        return hash(tuple(self.values))

    def __repr__(self):
        return f"{self.values}"


class SetSolver:
    def __init__(self, 
                 img: np.ndarray, 
                 model: torch.nn.Module,
                 out_path: str):
        self.img = img
        self.model = model
        
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        
        self.cards: List[DetectedCard] = [
            DetectedCard(img_bgr, rect) for img_bgr, rect in detect_cards(img)
        ]
        self._combined_tensor: torch.Tensor = self._get_combined_tensor()
        self.sets = []

    def _get_combined_tensor(self) -> torch.Tensor:
        """
        Get stacked tensor for all images of cards .
        """
        return torch.stack([(card._tensor_img) for card in self.cards])

    def _is_set(
        self, card1: DetectedCard, card2: DetectedCard, card3: DetectedCard
    ) -> bool:
        """
        Check if three cards form a set.

        Args:
            card1 (DetectedCard): First card.
            card2 (DetectedCard): Second card.
            card3 (DetectedCard): Third card.

        Returns:
            bool: True if the cards form a set, False otherwise.
        """
        coordinate_sums = list(
            (card1.values[i] + card2.values[i] + card3.values[i]) % 3 for i in range(4)
        )
        return coordinate_sums == [0, 0, 0, 0]
    

    def _present_set(self, set_indices, out_path: str) -> None:
        """
        Display the images of cards that form a set combined into one image.

        Args:
        set_indices (tuple): Indices of the cards that form a set.
        """
        card_imgs = [self.cards[idx].img_bgr for idx in set_indices]
        print([self.cards[idx].values for idx in set_indices])
        combined_img = cv2.hconcat(card_imgs)  
        cv2.imwrite(out_path, combined_img)
    
    def present_sets(self):
        """
        Display all sets of cards.
        """
        assert self.sets, "No sets found."
        for idx, set_indices in enumerate(self.sets):
            self._present_set(set_indices, f"{self.out_path}/set_{idx}.jpg")

    def predict_card_values(self) -> None:
        """
        Run model on card images to predict its values [number, color, shape, fill]
        # Note: Possibly could be done more 
        """
        out = self.model(self._combined_tensor)
        predicted_values = torch.argmax(out, dim=2).T
        for idx, card in enumerate(self.cards):
            card.values = predicted_values[idx].tolist()

    def find_sets(self) -> List[tuple]:
        """
        Find all sets of cards.

        Returns:
            list: List of tuples, each containing three indices of cards that form a set.
        """
        sets = []
        three_el_combinations = list(combinations(range(len(self.cards)), 3))
        for idx1, idx2, idx3 in three_el_combinations:
            if self._is_set(self.cards[idx1], self.cards[idx2], self.cards[idx3]):
                sets.append((idx1, idx2, idx3))
        self.sets = sets

    
        