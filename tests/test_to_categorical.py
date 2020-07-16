from keras_bed_sequence.utils import nucleotides_to_numbers, fast_to_categorical
import numpy as np


def test_to_categorical():
    nucleotides = "actg"
    expected = np.array([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0.25, 0.25, 0.25, 0.25],
        ]
    ], dtype=np.float64)
    encoding = nucleotides_to_numbers(
        nucleotides,
        np.array([
            ["a", "c", "n"],
            ["a", "a", "n"],
        ])
    )
    result = fast_to_categorical(encoding, len(nucleotides), 0.25)
    assert np.isclose(result, expected).all()
