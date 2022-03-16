from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, MaxPooling2D

SIDE = 28
IMAGE_SIZE = SIDE*SIDE

def get_generator(initial_dim, optimizer):
    generator = Sequential()

    generator.add(Dense(256, input_dim=initial_dim, kernel_initializer="random_normal"))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(IMAGE_SIZE, activation="sigmoid"))

    generator.compile(optimizer=optimizer, loss="binary_crossentropy")
    return generator


def get_generator_v2(initial_dim, optimizer):
    generator = Sequential()

    generator.add(Dense(IMAGE_SIZE, input_dim=initial_dim))
    generator.add(Reshape((SIDE, SIDE, 1), input_dim=IMAGE_SIZE))

    generator.add(Conv2DTranspose(filters=32, kernel_size=4, strides=1, padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(filters=16, kernel_size=4, strides=1, padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding="same", activation="tanh"))
    
    generator.compile(optimizer=optimizer, loss="binary_crossentropy")
    return generator