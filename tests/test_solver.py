import sys
import pytest
import numpy as np
from pathlib import Path

from typing import List

# add module root path
root_path = str(Path(".").resolve())
sys.path.append(str(Path(".").resolve()))

from src.solve import DetectedCard, SetSolver  # noqa: E402


@pytest.fixture
def cards_data() -> List[DetectedCard]:
    c1 = DetectedCard(
        np.zeros((100, 100, 3)), np.array([0, 0, 100, 100]), [1, 1, 1, 1]
    )
    c2 = DetectedCard(
        np.zeros((100, 100, 3)), np.array([0, 0, 100, 100]), [2, 2, 2, 2]
    )
    c3 = DetectedCard(
        np.zeros((100, 100, 3)), np.array([0, 0, 100, 100]), [3, 2, 1, 3]
    )
    c4 = DetectedCard(
        np.zeros((100, 100, 3)), np.array([0, 0, 100, 100]), [3, 3, 3, 3]
    )
    c5 = DetectedCard(
        np.zeros((100, 100, 3)), np.array([0, 0, 100, 100]), [1, 2, 3, 1]
    )
    return c1, c2, c3, c4, c5


def test_is_set(cards_data: List[DetectedCard]):
    solver = SetSolver(cards_data)

    # Test if three identical cards form a set
    c1, c2, c3, c4, _ = cards_data
    assert solver._is_set(c1, c2, c4) is True
    assert solver._is_set(c1, c3, c4) is False


def test_solver(cards_data: List[DetectedCard]):
    solver = SetSolver(cards_data)

    sets = solver.find_sets()
    assert len(sets) == 2
    assert (0, 1, 3) in sets
    assert (1, 2, 4) in sets