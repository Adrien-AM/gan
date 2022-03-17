from sre_constants import IN
from keras.models import Input, Model

def get_gan(discriminator, generator, initial_dim):
    # why discriminator and not generator ??
    #discriminator.trainable = False

    gan_input = Input(shape=(initial_dim,))

    # generate, then evaluate
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy")

    return gan