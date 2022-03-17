from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, BatchNormalization, UpSampling2D, MaxPooling2D

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

    kernel_size = 3

    # Image must be 28x28 -> 28=7*2*2, so 2 conv layers with 7 as base dim
    generator.add(Dense(1568, input_dim=initial_dim))
    generator.add(Reshape((7,7,32), input_dim=IMAGE_SIZE))

    generator.add(UpSampling2D())
    generator.add(Conv2D(filters=32, kernel_size=kernel_size, strides=1, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))

    generator.add(UpSampling2D())
    generator.add(Conv2D(filters=64, kernel_size=kernel_size, strides=1, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))
    generator.add(MaxPooling2D((2,2)))

    generator.add(UpSampling2D())
    generator.add(Conv2D(filters=16, kernel_size=kernel_size, strides=1, padding="same"))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding="same", activation="tanh"))
    
    generator.compile(optimizer=optimizer, loss="binary_crossentropy")
    return generator