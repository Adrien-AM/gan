import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
from keras.optimizers import adam_v2
from keras.models import load_model
from discriminator import get_discriminator
from gan import get_gan
from generator import get_generator
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import glob
import tensorflow as tf

os.environ['KERAS_BACKEND'] = "tensorflow"

tf.config.threading.set_intra_op_parallelism_threads(100)
tf.config.threading.set_inter_op_parallelism_threads(100)

SIDE = 200
IMAGE_SIZE = SIDE*SIDE*3
INITIAL_DIM = 1000

# copy pasted
def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, IMAGE_SIZE)
    
    return (x_train, y_train, x_test, y_test)

def load_vg_data():
    data = []

    print("Loading data...")
    if(os.path.exists("vg_data.npy")):
        data = np.load("vg_data.npy")
        #data = data.astype(np.float32)
    else:
        for filename in tqdm(glob.iglob("van_gogh_data/*/*Portrait*.jpg"), total=89): # 2025
            im = Image.open(filename).resize((SIDE, SIDE))
            data.append(np.asarray(im).reshape((IMAGE_SIZE)))
        data = np.array(data)
        #data = (data.astype(np.float32) - 127.5)/127.5

        np.save("vg_data", data)
    return data, None, None, None

def load_landscapes_data():
    data = []

    print("Loading data...")
    if(os.path.exists("land_data.npy")):
        data = np.load("land_data.npy")
        #data = data.astype(np.float32)
    else:
        for filename in tqdm(glob.iglob("landscapes/*.jpg"), total=4319): # 4319
            im = Image.open(filename).resize((SIDE, SIDE)).convert('RGB') # some pics are not detected as rgb ??
            data.append(np.asarray(im).reshape((IMAGE_SIZE)))
        data = np.array(data)
        #data = (data.astype(np.float32) - 127.5)/127.5

        np.save("land_data", data)
    return data, None, None, None

def train(epochs=1, batch_size=128):
    x_train, _, _, _ = load_landscapes_data()
    
    size = x_train.shape[0]
    x_train = x_train[:size]
    
    batch_count = int(x_train.shape[0] / batch_size)
    optimizer = adam_v2.Adam(learning_rate=0.0002, beta_1=0.5)

    discriminator = get_discriminator(optimizer)
    generator = get_generator(INITIAL_DIM, optimizer)
    gan = get_gan(discriminator, generator, INITIAL_DIM, optimizer)

    for epoch in range(1, epochs+1):
        print("Epoch n°%d" % epoch)
        for _ in tqdm(range(0, batch_count)):
            # random base for generator
            initial = np.random.normal(0, 1, size=[batch_size, INITIAL_DIM])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # generate images
            generated = generator.predict(initial)
            # mix with real images
            X = np.concatenate([image_batch, generated])
            # labels
            y = np.zeros(2*batch_size)
            y[:batch_size] = 0.9 # why not 1 ??

            # train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y)

            # train generator
            initial = np.random.normal(0, 1, size=[batch_size, INITIAL_DIM])
            y = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(initial, y)

    return generator


def generate_images(n, generator):
    initial = np.random.normal(0, 1, size=[n, INITIAL_DIM])
    pics = generator.predict(initial)

    pics = pics.reshape((n, SIDE, SIDE, 3))

    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(1, n, i+1)
        plt.imshow(pics[i], interpolation='nearest')
    plt.show(block=True)

def show_data(n):
    data, _, _, _ = load_landscapes_data()

    pics = np.array([data[np.random.randint(data.shape[0])] for i in range(n)])
    pics = pics.reshape((n, SIDE, SIDE, 3))

    fig = plt.figure()
    for i in range(n):
        fig.add_subplot(1, n, i+1)
        plt.imshow(pics[i], interpolation='nearest')
    plt.show(block=True)

if __name__ == "__main__":
    use_old = (len(sys.argv) > 1 and sys.argv[1] == "no-new")

    if use_old:
        generator = load_model("./generator.model")
    else:
        generator = train(20, 32)

    generate_images(3, generator)
    #show_data(3)

    if not use_old:
        generator.save("./generator.model")
    exit(0)
    
    