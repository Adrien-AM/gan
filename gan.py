from keras.models import Model
from keras.metrics import Mean
from keras.preprocessing.image import array_to_img
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

class GAN(Model):
    def __init__(self, disc, gen, initial, clip_value, lmbda):
        super().__init__()
        self.discriminator = disc
        self.generator = gen
        self.initial_dim = initial
        self.clip_value = clip_value
        self.g_loss = Mean(name="g_loss")
        self.d_loss = Mean(name="d_loss")
        self.lmbda = lmbda

    def compile(self, d_optimizer, g_optimizer, loss_function):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_function

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss]

    def train_step(self, data):
        real_images = data
        batch_size = tf.shape(real_images)[0]
        initial = tf.random.normal(shape=(batch_size, self.initial_dim))

        with tf.GradientTape(persistent=True) as tape:
            generated = self.generator(initial)

            disc_fake = self.discriminator(generated)
            disc_real = self.discriminator(real_images)

            alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1],
            minval=0, maxval=1.)

            diffs = generated - real_images
            interpolates = real_images + (alpha * diffs)
            gradients = tf.gradients(self.discriminator(interpolates), interpolates)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
            gp = tf.reduce_mean((slopes - 1.) ** 2)

            d_loss_real = tf.reduce_mean(disc_real)
            d_loss_fake = tf.reduce_mean(disc_fake)
            d_loss = -(d_loss_real - d_loss_fake) + self.lmbda * gp

            g_loss = d_loss_real - d_loss_fake

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        del tape

        return {"d_loss" : d_loss, "g_loss" : g_loss}


from keras.callbacks import Callback
import matplotlib.pyplot as plt

class GANMonitor(Callback):
    def __init__(self, initial, num_img=4):
        self.num_img = num_img
        self.initial_dim = initial
        self.seed = np.random.normal([16, initial])

    def on_epoch_end(self, epoch, logs=None):
        initials = tf.random.normal(shape=(self.num_img, self.initial_dim))
        generated = self.model.generator(initials)
        predictions = self.model.discriminator.predict(generated)
        generated.numpy()
        
        s = int(self.num_img/2)
        fig = plt.figure(figsize=(s, s))
        for i in range(self.num_img):
            plt.subplot(s, s, i+1)
            plt.imshow(array_to_img(generated[i]), cmap="gray")
            plt.title("%f" % predictions[i], pad=3.0, fontsize="small")
            plt.axis("off")
        plt.savefig('trace/epoch_{:03d}.png'.format(epoch + 1))
        plt.close(fig)
        #plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save("last_generator.h5")