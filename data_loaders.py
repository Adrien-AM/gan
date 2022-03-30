import numpy as np
from keras.datasets import mnist, fashion_mnist
from PIL import Image
from tqdm import tqdm
from glob import iglob
import os

SIDE = 160 # size must be divisible by 8
IMAGE_SIZE = SIDE*SIDE
CHANNELS = 3 # color channels, 3 for rgb
DATA_SIZE = 1200

def load_landscapes_data():
    data = []

    print("Loading landscapes data...", end="\t")
    if(os.path.exists("data/land_data.npy")):
        data = np.array(np.load("data/land_data.npy"), dtype=np.float32)
        data = data[:DATA_SIZE]
    else:
        i = 0
        for filename in tqdm(iglob("data/landscapes/*.jpg"), total=DATA_SIZE): # 4319
            im = Image.open(filename).resize((SIDE, SIDE)).convert('RGB') # some pics are not detected as rgb ??
            data.append(np.asarray(im))
            i += 1
            if i >= DATA_SIZE:
                break
        data = np.array(data)
        data = data.astype(np.float32)/255 # normalize to [0, 1]

        np.save("data/land_data", data)
    print("Done.")
    return data

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

def load_kana_data(datasize):
    print("Loading K49-MNIST data ...", end="\t")

    loader = np.load("data/kana-mnist/k49-train-imgs.npz")
    data = np.array(loader['arr_0'])[:DATA_SIZE]
    data = data.reshape(data.shape[0], SIDE, SIDE, 1).astype(np.float32)
    data = (data - 127.5) / 127.5
    np.random.shuffle(data)

    loader.close()

    print("Done.\nShape : ", data.shape)
    return data

import os
if __name__ == "__main__":
    os.system("rm data/*.npy")
    _ = load_landscapes_data()
