import numpy as np
from typing import List, Optional
from itertools import combinations


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
    def __init__(self, cards: List[DetectedCard] = []):
        self.cards = cards
        assert len(self.cards) == len(set(self.cards)), (
            "Found duplicates in cards. Make sure to provide unique cards in a list"
        )

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
        # TODO
        # a triple forms a set iff for each feature the sum of values is 0 mod 3
        for i in range(4): # check the 4 features (number, color, shape, fill) one by one
            if (card1.values[i] + card2.values[i] + card3.values[i]) % 3 != 0: # if for any feature the sum is not 0 mod 3 then it cannot be a set
                return False
        return True # if it passed the check for each feature then it is a set


    def find_sets(self) -> List[tuple]:
        """
        Find all sets of cards.

        Returns:
            list: List of tuples, each containing three indices of cards that form a set.
        """
        # TODO
        # finish this function
        found_sets = []
        n = len(self.cards)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if self._is_set(self.cards[i], self.cards[j], self.cards[k]):
                        found_sets.append((i, j, k))
        return found_sets
