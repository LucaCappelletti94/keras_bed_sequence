from keras_bed_sequence.utils import nucleotides_to_numbers
import numpy as np


def test_nucleotides_to_numbers():
    nucleotides = "actg"
    expected = np.array([
        [0, 1, 2, 2, 2, -1],
        [0, 1, 2, 2, 2, -1],
    ])
    result = nucleotides_to_numbers(
        nucleotides,
        np.array([
            ["a", "c", "t", "T", "t", "n"],
            ["a", "c", "t", "T", "t", "n"],
        ])
    )
    assert (expected == result).all()
