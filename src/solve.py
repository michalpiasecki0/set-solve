import numpy as np
from typing import List, Optional
from itertools import combinations

import torch
import torch.nn.functional as F
from src.constants import RUN_TRANSFORM
from src.segmentation import detect_cards


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
    def __init__(self, img: np.ndarray, model: torch.nn.Module):
        self.img = img
        self.model = model

        self.cards: List[DetectedCard] = [
            DetectedCard(img_bgr, rect) for img_bgr, rect in detect_cards(img)
        ]
        self._combined_tensor: torch.Tensor = self._get_combined_tensor()
        # self.tensor_for_net = self._get_combined_tensor()
        self.sets = []


    def predict_card_values(self) -> None:
        print(self._combined_tensor.shape)
        out = self.model(self._combined_tensor)
        predicted_values = torch.argmax(out, dim=2).T
        for idx, card in enumerate(self.cards):
            card.values = predicted_values[idx].tolist()
            print(card.values)

    def _get_combined_tensor(self):
        """
        Get internal representation of cards for set solving.
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
        return sets
