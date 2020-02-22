import os
from keras_bed_sequence import BedSequence
from ucsc_genomes_downloader import Genome


def test_bed_sequence():
    batch_size = 32
    bed_sequence = BedSequence(
        Genome("hg19", chromosomes="chr1"),
        "{cwd}/test.bed".format(
            cwd=os.path.dirname(os.path.abspath(__file__))
        ),
        batch_size
    )
    x = bed_sequence[0]
    assert (bed_sequence[0] == bed_sequence[0]).all()
    assert x.shape == (32, 200, 4)
