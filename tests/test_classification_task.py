import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras_mixed_sequence import MixedSequence
from keras_bed_sequence import BedSequence


def build_model():
    model = Sequential([
        Flatten(),
        Dense(1)
    ])
    model.compile(
        optimizer="nadam",
        loss="MSE"
    )
    return model


def test_classification_task():
    batch_size = 32
    bed_sequence = BedSequence(
        "hg19",
        "{cwd}/test.bed".format(
            cwd=os.path.dirname(os.path.abspath(__file__))
        ),
        batch_size
    )
    assert 200 == bed_sequence.window_length
    y = np.random.randint(
        2,
        size=(bed_sequence.samples_number, 1)
    )
    mixed_sequence = MixedSequence(
        x=bed_sequence,
        y=y,
        batch_size=batch_size
    )
    model = build_model()
    model.fit_generator(
        mixed_sequence,
        steps_per_epoch=mixed_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
