import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Reshape, Flatten, Conv2DTranspose, Input
from keras_mixed_sequence import MixedSequence
from keras_bed_sequence import BedSequence


def build_model():
    inputs = Input(shape=(200, 4))

    flattened = Flatten()(inputs)

    output1 = Dense(
        units=1,
        activation="relu",
        name="output1"
    )(flattened)

    hidden = Reshape((200, 4, 1))(inputs)
    hidden = Conv2D(16, kernel_size=3, activation="relu")(hidden)
    hidden = Conv2DTranspose(1, kernel_size=3, activation="relu")(hidden)
    output2 = Reshape((200, 4), name="output2")(hidden)

    model = Model(
        inputs=inputs,
        outputs=[output1, output2],
        name="my_model"
    )

    model.compile(
        optimizer="nadam",
        loss="MSE"
    )

    return model


def test_bed_sequence():
    batch_size = 32
    bed_sequence = BedSequence(
        "hg19",
        "{cwd}/test.bed".format(
            cwd=os.path.dirname(os.path.abspath(__file__))
        ),
        batch_size
    )
    y = np.random.randint(
        2,
        size=(bed_sequence.samples_number, 1)
    )
    mixed_sequence = MixedSequence(
        bed_sequence,
        {
            "output1": y,
            "output2": bed_sequence
        },
        batch_size
    )
    model = build_model()
    model.fit_generator(
        mixed_sequence,
        steps_per_epoch=mixed_sequence.steps_per_epoch,
        epochs=2,
        verbose=0,
        shuffle=True
    )
