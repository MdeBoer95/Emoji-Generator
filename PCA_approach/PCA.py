import cv2
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
from skimage import restoration
import scipy
from scipy import ndimage

def load_data(path=os.getcwd()+'/../emoji_data/emojis_jpg_root/all_emojis'):
    emojis_r = np.zeros((64*64, len(listdir(path))))
    emojis_g = np.zeros((64*64, len(listdir(path))))
    emojis_b = np.zeros((64*64, len(listdir(path))))
    gray = np.zeros((64*64, len(listdir(path))))

    j=0
    for i in listdir(path):
        img = Image.open(join(path, i))
        img1 = img.convert('L')
        img = np.array(img)

        if(len(img.shape) == 2): # I think not needed, but is here for secureness
            continue
        
        emojis_r[:, j] = img[:,:, 0].flatten()
        emojis_g[:, j] = img[:, :, 1].flatten()
        emojis_b[:, j] = img[:, :, 2].flatten()
        gray[:, j] = np.array(img1).flatten()
        j += 1
    


    return emojis_r, emojis_g, emojis_b, gray







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

# Works only between 0 and 1
def eigenface(img, n_components):
    img = img.T / 255
    pca = PCA(n_components)
    pca = pca.fit(img)
    eigen_value = pca.explained_variance_  
    face = pca.components_
    return eigen_value, face,pca


########################################## Newly added Stuff #################################################


def load_data_all_color(path=os.getcwd()+'/emoji_data/emojis_jpg_root/all_emojis'):
    emojis = np.zeros((64*64*3, len(listdir(path))))

    j=0
    for i in listdir(path):
        img = Image.open(join(path, i))
        img1 = img.convert('L')
        img = np.array(img)

        if(len(img.shape) == 2): # I think not needed, but is here for secureness
            print("It was continued.")
            continue
        
        emojis[:, j] = img.flatten()
        j += 1
    return emojis



def show_eigenface_all_colors(vectors, component):
    plt.figure()
    for i in range(component):
        #print(np.std(vectors[i,:])*255*20)
        print(np.max(vectors[i,:]))
        print(np.min(vectors[i,:]))
        plt.subplot(5, 6, i+1)
        plt.imshow((vectors[i,:].reshape(64,64,3)+ 0.05)*10)
    plt.show()



def eigenvalue_range(data,pca,components):
    a = np.zeros((components,453))
    for i in range(data.shape[1]):
        a[:,i] = pca.transform(data[:,i].reshape(1,-1)/255)
    
    b = np.mean(a,axis=1)
    c = np.std(a,axis=1)
    #print(b.shape)
    #print(b)
    #print(c)
    return c


def all_color_pca():

    # PCA + Show Eigenvectors
    components = 120
    emojis = load_data_all_color()
    e_values, e_vectors,pca = eigenface(emojis, components)
    print(len(pca.explained_variance_ratio_))
    print(sum(pca.explained_variance_ratio_))
    #show_eigenface_all_colors(e_vectors,components)
    
    #print(np.min(e_vectors))
    #print(np.min(np.abs(e_vectors)))

    eigenvalue_range(emojis,pca,components)

    return 
    ############ Sanity Test ############

    # Transform picture to eigenvalues
    #print(emojis.shape)
    temp = pca.transform(emojis[:,0].reshape(1,-1)/255)
    print(temp)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(temp.reshape(-1), label="G")
    plt.legend()
    plt.gcf()
    plt.show()

    # Get Mean of Pictures and 
    mean_face = np.mean(emojis,axis=1)/255
    #print("this now")
    #print(temp)
    # Reconstruct image
    res_arr = mean_face 
    for i in range(len(temp[0,:])):
        res_arr += temp[0,i]*e_vectors[i,:]

    print(np.min(res_arr))
    print(np.max(res_arr))
    print(res_arr[res_arr<0].shape)
    print(res_arr[res_arr>1].shape)
    
    #temp = pca.inverse_transform(temp.reshape(1,-1))
    res_arr = res_arr.reshape(64,64,3)

    plt.figure()
    plt.subplot(1, 4,1)
    plt.imshow((res_arr))
    plt.subplot(1, 4,2 )
    res_arr = restoration.denoise_nl_means(res_arr,fast_mode=False)
    plt.imshow((res_arr))

    plt.subplot(1,4,3)

    sharpened = np.zeros((64,64,3))
    for i in range(3):
        #blurred_f = ndimage.gaussian_filter(res_arr[:,:,i], 3)

        filter_blurred_f = ndimage.gaussian_filter(res_arr[:,:,i], 1)

        alpha = 2
        sharpened[:,:,i] = res_arr[:,:,i] + alpha * (res_arr[:,:,i] - filter_blurred_f)

    plt.subplot(1,4,3)
    plt.imshow(sharpened)

    plt.subplot(1,4,4)
    plt.imshow(emojis[:,0].reshape(64,64,3)/255)



    plt.show()

def smooth_and_sharp(image, alpha=2,fast_mode=False):
    res_arr = restoration.denoise_nl_means(image,fast_mode=fast_mode)
    sharpened = np.zeros((64,64,3))
    for i in range(3):
        filter_blurred_f = cv2.medianBlur(res_arr[:,:,i], 5)
        sharpened[:,:,i] = res_arr[:,:,i] + alpha * (res_arr[:,:,i] - filter_blurred_f)
    return sharpened



def show_gray_eigenfaces(vector_gr):
    plt.figure()
    for i in range(20):
        #print(np.std(vectors[i,:])*255*20)
        print(np.max(vector_gr[i,:]))
        print(np.min(vector_gr[i,:]))
        plt.subplot(4, 5, i+1)
        plt.imshow((vector_gr[i,:].reshape(64,64)+ 0.05)*10)
    plt.show()


def color_seperated_pca():
    emoji_r, emoji_g, emoji_b, gray = load_data()   
    value_r, vector_r,pca = eigenface(emoji_r, 20)
    value_g, vector_g,pca = eigenface(emoji_g, 20)
    value_b, vector_b,pca = eigenface(emoji_b, 20)
    value_gr, vector_gr,pca = eigenface(gray, 20)

    show_gray_eigenfaces(vector_gr)

    show_eigenface(vector_r, vector_g, vector_b, 20)


if __name__ == "__main__":
    #color_seperated_pca()
    all_color_pca()


    # vector_gr = vector_gr*255*20
    # plt.figure()
    # for i in range(20):
    #     plt.subplot(4, 5, i+1)
    #     img = Image.fromarray(vector_gr[i].reshape(64, 64))
    #     plt.imshow(img)
    # plt.show()
    print(11)