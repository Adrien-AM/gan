# -*- coding: utf-8 -*-
"""GAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lRlTXY5Qo2f-YY4Egj_L4nt-TNcfi5mW
"""

import os
# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %tensorflow_version 2.xs
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import (BatchNormalization, Conv2D, Dense, LeakyReLU, ReLU, Dropout, Flatten,
                          Reshape)
# Commented out IPython magic to ensure Python compatibility.
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Sequential, Input, Model, load_model
from keras.initializers.initializers_v2 import RandomNormal 
from keras.datasets import mnist, fashion_mnist
from keras.optimizer_v2 import adam
from keras.losses import BinaryCrossentropy
from plot_keras_history import show_history
from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import np_config
from gan import GAN, GANMonitor, BATCH_SIZE

# Numpy functions for tensorflow
np_config.enable_numpy_behavior()

SIDE = 28
IMAGE_SIZE = SIDE*SIDE
INITIAL_DIM = 10
CHANNELS = 1 # color channels, 1 for grayscale
DATA_SIZE = 20000

def get_generator_v2():
    generator = Sequential(name="generator")

    weight_initializer = RandomNormal(0, 0.02)
    kernel_size = 4

    # Image must be 28x28 -> 28=7*2*2, so 2 conv layers with 7 as base dim
    generator.add(Dense(7 * 7 * 256, input_dim=INITIAL_DIM))
    generator.add(BatchNormalization())
    generator.add(ReLU(0.2))

    generator.add(Reshape((7, 7, 256)))

    generator.add(Conv2DTranspose(filters=64, kernel_size=kernel_size, strides=2, padding="same", kernel_initializer=weight_initializer))
    generator.add(BatchNormalization())
    generator.add(ReLU(0.2))

    generator.add(Conv2DTranspose(filters=128, kernel_size=kernel_size, strides=2, padding="same", kernel_initializer=weight_initializer))
    generator.add(BatchNormalization())
    generator.add(ReLU(0.2))

    generator.add(Conv2D(filters=CHANNELS, kernel_size=kernel_size, padding="same", activation="tanh"))
    
    return generator

def get_discriminator_v2():
    discriminator = Sequential(name="discriminator")

    kernel_size = 4

    discriminator.add(Conv2D(filters=64, kernel_size=kernel_size, strides=2, padding="same", input_shape=(SIDE, SIDE, CHANNELS)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=64, kernel_size=kernel_size, strides=2, padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))    

    discriminator.add(Flatten())
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation="sigmoid"))

    return discriminator


def load_minst_data():
    print("Loading MNIST data ...", end="\t")
    # load the data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = x_train.reshape(x_train.shape[0], SIDE, SIDE, 1).astype(np.float32)
    x_train = (x_train - 127.5) / 127.5
    np.random.shuffle(x_train) # necessary ?

    print("Done.")
    return (x_train[:DATA_SIZE], y_train, x_test, y_test)

def load_kana_data():
    print("Loading K49-MNIST data ...", end="\t")

    loader = np.load("data/kana-mnist/k49-train-imgs.npz")
    data = np.array(loader['arr_0'])[:DATA_SIZE]
    data = data.reshape(data.shape[0], SIDE, SIDE, 1).astype(np.float32)
    data = (data - 127.5) / 127.5
    np.random.shuffle(data)

    loader.close()

    print("Done.\nShape : ", data.shape)
    return data

def train_v2(epochs, data):
    discriminator = get_discriminator_v2()
    generator = get_generator_v2()

    if "-v" in sys.argv:
        discriminator.summary()
        generator.summary()
    LR = 0.0002

    gan = GAN(disc=discriminator, gen=generator, initial=INITIAL_DIM)
    gan.compile(
        d_optimizer=adam.Adam(LR, beta_1=0.5),
        g_optimizer=adam.Adam(LR, beta_1=0.5),
        loss_function=BinaryCrossentropy(label_smoothing=0.05)
    )

    history = gan.fit(x=data, epochs=epochs, batch_size=BATCH_SIZE, callbacks=[GANMonitor(num_img=4, initial=INITIAL_DIM)])
    return gan, history

def generate_images(n, generator, discriminator):
    # Generate some pictures from random noise
    initial = np.random.normal(size=[n, INITIAL_DIM])
    pics = generator.predict(initial)
    # Check what discriminator says
    is_good = discriminator.predict(pics)

    pics = pics.reshape((n, SIDE, SIDE))

    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(1, n, i+1, title="Discrim : %f" % is_good[i])
        plt.imshow(pics[i], interpolation='nearest', cmap="gray")
    plt.show(block=True)


def show_data(data, n=3):
    pics = np.array([data[np.random.randint(data.shape[0])] for _ in range(n)])
    pics = pics.reshape((n, SIDE, SIDE))

    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(1, n, i+1)
        plt.imshow(pics[i], interpolation='nearest', cmap="gray")
    plt.show(block=True)


def test_discriminator(data, discriminator, generator, n=3):
    pics = np.array([data[np.random.randint(data.shape[0])] for _ in range(n)])
    prediction = discriminator.predict(pics)
    # Should be 1
    print("Average prediction error for real data : ", np.average(abs(prediction-1)))

    fakes = np.random.rand(n, SIDE, SIDE)
    prediction = discriminator.predict(fakes)
    # Should be 0
    print("Average prediction error for random fakes : ", np.average(prediction))

    generated = generator.predict(np.random.normal(size=(n, INITIAL_DIM)))
    prediction = discriminator.predict(generated)
    # Should be 0
    print("Average prediction error for random generated : ", np.average(prediction))

if __name__ == "__main__":
    print("Usage : \n -n to use existing GAN\n -v to display NN layers")
    epochs = 30

    data = load_kana_data()

    new = "-n" not in sys.argv

    if new:
        print("Creating new GAN from scratch")
        gan, history = train_v2(epochs, data)
    else:
        print("Using old discriminator and generator")
        try:
            generator = load_model("last_generator.h5")
            discriminator = load_model("last_discriminator.h5")
        except IOError:
            print("Error loading files. Check last_generator.h5 and last_discriminator.h5 exist in current dir.")
            exit(1)

        gan = GAN(disc=discriminator, gen=generator, initial=INITIAL_DIM)

    generate_images(4, gan.generator, gan.discriminator)
    if new:
        show_history(history)
        plt.show()

    if input("Save generator ? ([y]/n)") != "n":
        gan.generator.save("last_generator.h5")
    if input("Save discriminator ? ([y]/n)") != "n":
        gan.discriminator.save("last_discriminator.h5")

    exit(0)

 
