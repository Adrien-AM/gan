from keras.layers import Dense, LeakyReLU, Dropout
from keras.models import Model, Sequential

SIDE = 200
IMAGE_SIZE = SIDE*SIDE*3

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