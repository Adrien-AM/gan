import sys
from turtle import shape
from keras.models import Input, Model
from keras.metrics import Mean
from keras.preprocessing.image import array_to_img
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32

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


class GAN(Model):
    def __init__(self, disc, gen, initial):
        super().__init__()
        self.discriminator = disc
        self.generator = gen
        self.initial_dim = initial
        self.g_loss = Mean(name="g_loss")
        self.d_loss = Mean(name="d_loss")

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

        generated = self.generator(initial)
        combined = tf.concat([generated, real_images], axis=0)
        labels = tf.concat([tf.zeros((batch_size, 1)), 
                            tf.ones((batch_size, 1))], axis=0)

        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        initial = tf.random.normal(shape=(batch_size, self.initial_dim))
        labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(initial))
            g_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss" : d_loss, "g_loss" : g_loss}
'''
    @tf.function
    def train_step(self, data):
        batch_size = BATCH_SIZE        
        initial = tf.random.normal(shape=(batch_size, self.initial_dim))

        # TRAIN DISCRIMINATOR
        with tf.GradientTape() as tape:
            prediction = self.discriminator(data, training=True)
            labels = tf.ones((batch_size, 1), dtype=np.float32)
            d_loss_real = self.loss_fn(labels, prediction) # prediction result on real data

            fakes = self.generator(initial)
            prediction = self.discriminator(fakes, training=True)
            labels = tf.zeros((batch_size, 1), dtype=np.float32)
            d_loss_fake = self.loss_fn(labels, prediction)

            total_d_loss = (d_loss_fake + d_loss_real) / 2

            
        grads = tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # TRAIN GENERATOR
        labels = tf.ones((batch_size, 1))
        initial = tf.random.normal(shape=(batch_size, self.initial_dim))

        with tf.GradientTape() as tape:
            fakes = self.generator(initial, training=True)
            prediction = self.discriminator(fakes, training=False) # True ?s
            g_loss = self.loss_fn(labels, prediction)

        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.d_loss.update_state(total_d_loss)
        self.g_loss.update_state(g_loss)
        
        return {"d_loss": self.d_loss.result(), "g_loss": self.g_loss.result()}
'''

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