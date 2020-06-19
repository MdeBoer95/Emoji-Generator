from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

def load_data(path=os.getcwd()+'/../emoji_data/emojis_root/all_emojis'):
    emojis_r = np.zeros((64*64, len(listdir(path))))
    emojis_g = np.zeros((64*64, len(listdir(path))))
    emojis_b = np.zeros((64*64, len(listdir(path))))
    gray = np.zeros((64*64, len(listdir(path))))

    j=0
    for i in listdir(path):
        img = Image.open(join(path, i))
        img1 = img.convert('L')
        img = np.array(img)
        if(len(img.shape) == 2):
            continue
        
        emojis_r[:, j] = img[:,:, 0].flatten()
        emojis_g[:, j] = img[:, :, 1].flatten()
        emojis_b[:, j] = img[:, :, 2].flatten()
        gray[:, j] = np.array(img1).flatten()
        j += 1
        
    return emojis_r, emojis_g, emojis_b, gray

def eigenface(img, n_components):
    img = img.T / 255
    pca = PCA(n_components)
    pca = pca.fit(img)
    eigen_value = pca.explained_variance_
    face = pca.components_
    return eigen_value, face


def combine_RGB(vector_r, vector_g, vector_b, face_i):
    rgb = np.zeros((64, 64, 3), 'uint8')
    rgb[:, :, 0] = vector_r[face_i].reshape((64, 64))*255
    rgb[:, :, 1] = vector_g[face_i].reshape((64, 64))*255
    rgb[:, :, 2] = vector_b[face_i].reshape((64, 64))*255
    return rgb

def show_eigenface(vector_r, vector_g, vector_b, component):
    plt.figure()
    for i in range(component):
        rgb = combine_RGB(vector_r, vector_g, vector_b, i)
        img = Image.fromarray(rgb*20, 'RGB')
        plt.subplot(4, 5, i+1)
        plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    emoji_r, emoji_g, emoji_b, gray = load_data()
    value_r, vector_r = eigenface(emoji_r, 20)
    value_g, vector_g = eigenface(emoji_g, 20)
    value_b, vector_b = eigenface(emoji_b, 20)
    value_gr, vector_gr = eigenface(gray, 20)
    show_eigenface(vector_r, vector_g, vector_b, 20)
    # vector_gr = vector_gr*255*20
    # plt.figure()
    # for i in range(20):
    #     plt.subplot(4, 5, i+1)
    #     img = Image.fromarray(vector_gr[i].reshape(64, 64))
    #     plt.imshow(img)
    # plt.show()
    print(11)