from keras.layers import Dense, LeakyReLU, Dropout, Conv2D, Flatten, BatchNormalization, Reshape
from keras.models import Sequential

SIDE = 28
IMAGE_SIZE = SIDE*SIDE

def get_discriminator(optimizer):
    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=IMAGE_SIZE, kernel_initializer="random_normal"))
    discriminator.add(LeakyReLU(0.2)) # activation function
    discriminator.add(Dropout(0.3)) # avoid overfitting

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2)) # activation function
    discriminator.add(Dropout(0.3)) # avoid overfitting

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2)) # activation function
    discriminator.add(Dropout(0.3)) # avoid overfitting

    discriminator.add(Dense(1, activation="sigmoid")) # output between 0 and 1
    discriminator.compile(optimizer=optimizer, loss="binary_crossentropy")

    return discriminator

def get_discriminator_v2(optimizer):
    discriminator = Sequential()

    kernel_size = 3

    discriminator.add(Reshape((SIDE, SIDE, 1), input_shape=(SIDE, SIDE)))

    discriminator.add(Conv2D(filters=64, kernel_size=kernel_size, strides=2))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=128, kernel_size=kernel_size, strides=2))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters=32, kernel_size=kernel_size, strides=2))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dropout(0.2))

    discriminator.add(Dense(1, activation="sigmoid"))

    discriminator.compile(optimizer=optimizer, loss="binary_crossentropy")
    return discriminator

''' INTERESTING
# DCGAN discriminator
def get_discriminator():
    image_input = keras.Input(shape=(image_size, image_size, 3))
    x = image_input
    for _ in range(depth):
        x = layers.Conv2D(
            width, kernel_size=4, strides=2, padding="same", use_bias=False,
        )(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU(alpha=leaky_relu_slope)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)
    output_score = layers.Dense(1)(x)

    return keras.Model(image_input, output_score, name="discriminator")'''