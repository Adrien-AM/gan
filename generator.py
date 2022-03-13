from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Reshape

SIDE = 200
IMAGE_SIZE = SIDE*SIDE*3

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
