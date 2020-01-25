from typing import Dict, Tuple, Union
from tensorflow.keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np
from ucsc_genomes_downloader import Genome
from tqdm.auto import tqdm
from keras_mixed_sequence.utils import sequence_length, batch_slice


class BedSequence(Sequence):

    def __init__(
        self,
        assembly: str,
        bed: Union[pd.DataFrame, str],
        batch_size: int = 32,
        verbose: bool = True,
        nucleotides: str = "actg",
        genome_kwargs: Dict = None
    ):
        """Return new BedGenerator object.

        Parameters
        --------------------
        assembly: str,
            Genomic assembly from ucsc from which to extract sequences.
            For instance, "hg19", "hg38" or "mm10".
        bed: Union[pd.DataFrame, str],
            Either path to file or Pandas DataFrame containing minimal bed columns,
            like "chrom", "chromStart" and "chromEnd".
        batch_size: int = 32,
            Batch size to be returned for each request.
            By default is 32.
        verbose: bool = True,
            Whetever to show a loading bar.
        nucleotides: str = "actg",
            Nucleotides to consider when one-hot encoding.
        genome_kwargs: Dict = None,
            Parameters to pass to the Genome object.

        Returns
        --------------------
        Return new BedGenerator object.
        """
        # If the given bed file is provided
        # we load the file using pandas.
        if isinstance(bed, str):
            bed = pd.read_csv(bed, sep="\t")

        # We retrieve the required chromosomes
        # from the required assembly.
        self._genome = Genome(
            assembly=assembly,
            chromosomes=bed.chrom.unique(),
            verbose=verbose,
            **({} if genome_kwargs is None else genome_kwargs)
        )

        self._batch_size, self._nucleotides = batch_size, nucleotides

        # We extract the sequences of the bed file from
        # the given genome.
        sequences = self._genome.bed_to_sequence(bed).sequence

        self._x = np.array([
            [
                self._nucleotides.find(nucleotide)
                for nucleotide in sequence.lower()
            ] for sequence in tqdm(
                sequences,
                desc="Converting nucleotides to numeric classes",
                disable=not verbose
            )
        ], dtype=np.int8)

    def on_epoch_end(self):
        """Shuffle private bed object on every epoch end."""
        np.random.shuffle(self._x)

    def __len__(self) -> int:
        """Return length of bed generator."""
        return sequence_length(self._x, self._batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.

        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.

        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding to given batch index.
        """
        return to_categorical(
            self._x[batch_slice(idx, self._batch_size)],
            num_classes=len(self._nucleotides)
        )
