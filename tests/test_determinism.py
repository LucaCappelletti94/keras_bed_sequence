import numpy as np
import pandas as pd
from keras_mixed_sequence import MixedSequence, VectorSequence
from keras_bed_sequence import BedSequence
from crr_labels import fantom
from ucsc_genomes_downloader import Genome
from tqdm.auto import trange, tqdm
import os


def test_genomic_sequence_determinism():
    batch_size = 32
    epochs = 5
    enhancers = pd.read_csv("tests/enhancers.csv")
    promoters = pd.read_csv("tests/promoters.csv")

    genome = Genome("hg19", chromosomes=["chr1"])
    for region in tqdm((enhancers, promoters), desc="Region types"):
        y = np.arange(0, len(region), dtype=np.int64)
        mixed_sequence = MixedSequence(
            x=BedSequence(genome, region, batch_size),
            y=VectorSequence(y, batch_size)
        )
        reference_mixed_sequence = MixedSequence(
            x=BedSequence(genome, region, batch_size=len(region), shuffle=False),
            y=VectorSequence(y, batch_size=len(region), shuffle=False)
        )
        X, _ = reference_mixed_sequence[0]
        for _ in trange(epochs, desc="Epochs", leave=False):
            for step in range(mixed_sequence.steps_per_epoch):
                xi, yi = mixed_sequence[step]
                assert (X[yi.astype(int)] == xi).all()
            mixed_sequence.on_epoch_end()
