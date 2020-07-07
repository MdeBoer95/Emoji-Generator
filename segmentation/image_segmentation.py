import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import skimage.segmentation as seg
from os.path import join, exists
import os

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap=cmap)
    ax.axis('off')
    return fig, ax

"""
def snake_ellipse(aimage):
    fig, ax = image_show(aimage)
    radians = np.linspace(0, 2 * np.pi, 32)
    c = 32 + 28 * np.cos(radians)  # polar co-ordinates # 32 is x center, 30 is x radius
    r = 27 + 11 * np.sin(radians)   # 16 is y center
    ellipse = np.array([c, r]).T
    ax.plot(ellipse[:, 0], ellipse[:, 1], '--r', lw=3)
    snake = seg.active_contour(aimage, ellipse)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    return snake
"""

parts = {"ears":(0, 16), "eyes":(0, 42), "cheeks":(30, 46), "mouth":(42, 64)}

images = io.ImageCollection('emojis_jpg_root/all_emojis/*.jpg')

"""
snake = snake_ellipse(image)
plt.show()
cropped = image[0:32, 0:64]
# Shows the image in image viewer
image_show(cropped)
"""
"""
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = canny(image / 255.)
image_show(edges)
# image_show(image)
fill_coins = ndi.find_objects(edges)
"""

'''
for part, ind in parts.items():
    print(part)
    for image, name in zip(images, images.files):
        image_felzenszwalb = seg.felzenszwalb(image[ind[0]:ind[1]], scale=64 * 64 / 5)    # Important hyperparam
        image_felzenszwalb_rgb = color.label2rgb(image_felzenszwalb, image[ind[0]:ind[1]], kind='avg')
        #image_show(image_felzenszwalb_rgb)
        #plt.show()
        #bounds = seg.find_boundaries(image_felzenszwalb)
        n_clust = np.unique(image_felzenszwalb).size
        #print(n_clust)
        for x in range(1, n_clust):
            segind = np.where(image_felzenszwalb == x)
            # TODO: Save color
            # TODO: How to target specific parts, like eyes?
            # TODO: Deal with wechat emojis!
            if segind[0].max()-segind[0].min() > 48 or segind[1].max()-segind[1].min() > 48:
                continue
            aseg = np.zeros((ind[1]-ind[0],64))
            aseg[segind] = image_felzenszwalb[segind]
            aseg_rgb = color.label2rgb(aseg, image[ind[0]:ind[1]], kind='avg')
            img_name = name.split("/")[-1].split(".")[0]
            plt.imsave('segmented_data/'+part+"/data/"+img_name+"_"+str(x)+".jpg", aseg_rgb, cmap=plt.cm.gray)
'''
def read_labelmap(labelfile):
    with open(labelfile) as lfile:
        # skip first line
        next(lfile)
        labelmap = {}
        for line in lfile:
            label, color, _, _ = line.split(':')
            labelmap[label] = tuple(int(c) for c in color.split(','))
        return labelmap


def extract(image, mask, color):
    segment = np.zeros(mask.shape,dtype=np.uint8)
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            segment[i,j,:] = image[i,j,:] if np.all(mask[i,j,:] == np.array(color)) else np.array([0,0,0])
    return segment


def extract_segments(annoation_folder, image_folder, output_dir="segmentation_data"):
    """
    Extract the different segments from annotated images and store them in different folders
    :param annoation_folder: root folder of exported images in SegmentationMask1.1 format i.e. the parent directory of
    the SegmentationClass directory
    :param image_folder: folder that contains the corresponding images. Images are expected to be in .jpg format
    """
    if not exists(output_dir):
        os.mkdir(output_dir)
    labelmap = read_labelmap(join(annoation_folder, "labelmap.txt"))
    masks = io.ImageCollection(join(annoation_folder,"SegmentationClass", "*.png"))
    images = io.ImageCollection(join(image_folder, "*.jpg"))
    counts = {label : 0 for label in labelmap.keys()}
    for image, mask in zip(images, masks):
        for label, color in labelmap.items():
            segment = extract(image, mask, color)
            # Save segment in corresponding folder
            segmentfolder = join(output_dir, label)
            if not np.all(segment == 0):
                if not exists(segmentfolder):
                    os.mkdir(segmentfolder)
                io.imsave(join(segmentfolder, str(counts[label]) + ".jpg"), segment)
                counts[label] += 1


if __name__ == '__main__':
    extract_segments(annoation_folder="/home/marcel/Uni/Master/3.Semester/DGM/Emoji-Gen/annotations",
                     image_folder="/home/marcel/Uni/Master/3.Semester/DGM/Emoji-Gen/fb_jpg")