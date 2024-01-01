import tensorflow as tf
from keras import layers, models


def build_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

