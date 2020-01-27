import os
import pytest
from keras_bed_sequence import BedSequence


def test_misuse():
    """Testing wrong parameterization."""
    with pytest.raises(ValueError):
        BedSequence(
            "hg19",
            "{cwd}/misuse.bed".format(
                cwd=os.path.dirname(os.path.abspath(__file__))
            ),
            32
        )
