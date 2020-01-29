import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Reshape, Conv2DTranspose
from keras_mixed_sequence import MixedSequence
from keras_bed_sequence import BedSequence


def build_model():
    model = Sequential([
        Reshape((200, 4, 1)),
        Conv2D(16, kernel_size=3, activation="relu"),
        Conv2DTranspose(1, kernel_size=3, activation="relu"),
        Reshape((-1, 200, 4))
    ])
    model.compile(
        optimizer="nadam",
        loss="MSE"
    )
    return model


def test_model_autoencoder():
    batch_size = 32
    bed_sequence = BedSequence(
        "hg19",
        "{cwd}/test.bed".format(
            cwd=os.path.dirname(os.path.abspath(__file__))
        ),
        batch_size
    )
    mixed_sequence = MixedSequence(bed_sequence, bed_sequence, batch_size)
    assert bed_sequence.steps_per_epoch == mixed_sequence.steps_per_epoch
    model = build_model()
    model.fit_generator(
        mixed_sequence,
        steps_per_epoch=mixed_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
