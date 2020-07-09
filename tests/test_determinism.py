import numpy as np
from keras_mixed_sequence import MixedSequence
from keras_bed_sequence import BedSequence
from crr_labels import fantom
from ucsc_genomes_downloader import Genome
from tqdm.auto import trange, tqdm

def test_simple_determinism():
    classes = 10
    number = 100000
    epochs = 100
    x = np.arange(0, number, dtype=np.int64)
    y = np.random.randint(0, classes, size=number)

    ms = MixedSequence(x, y, batch_size=100)

    for _ in range(epochs):
        for step in range(ms.steps_per_epoch):
            xi, yi = ms[step]
            assert (y[xi.astype(int)] == yi).all()
        ms.on_epoch_end()


def test_genomic_sequence_determinism():
    batch_size = 32
    epochs = 100
    cell_line = "GM12878"
    enhancers, promoters = fantom(
        # list of cell lines to be considered.
        cell_lines=[cell_line],
        window_size=200,  # window size to use for the various regions.
    )
    genome = Genome("hg19")
    for region in tqdm((enhancers, promoters), desc="Region types"):
        bed_sequence = BedSequence(
            genome,
            region,
            batch_size
        )
        y = np.arange(0, len(region), dtype=np.int64)
        mixed_sequence = MixedSequence(
            x=bed_sequence,
            y=y,
            batch_size=batch_size
        )
        reference_bed_sequence = BedSequence(
            genome,
            region,
            len(y)
        )
        reference_mixed_sequence = MixedSequence(
            x=reference_bed_sequence,
            y=y,
            batch_size=len(y)
        )
        X, _ = reference_mixed_sequence[0]
        for _ in trange(epochs, desc="Epochs", leave=False):
            for step in range(mixed_sequence.steps_per_epoch):
                xi, yi = mixed_sequence[step]
                assert (X[yi.astype(int)] == xi).all()
            mixed_sequence.on_epoch_end()