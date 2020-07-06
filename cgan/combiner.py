import cv2
from PIL import Image, ImageDraw
import random
import numpy as np

from cgan.dataloading import EmojiFaceSegments
from PCA_approach.PCA import smooth_and_sharp


def combine_parts(parts_list, nogan=True):
    dataset_path = "../segmentation/segmentation_data" if nogan else "inference_results"

    # Create background
    background = create_background()

    # Load dataset
    dataset = EmojiFaceSegments(dataset_path)

    # Choose parts
    for apart in parts_list:
        curr_parts = list(map(lambda x: x[0], filter(lambda x: x[1]==apart, dataset.data)))
        # Pick random part
        pick = random.choice(curr_parts).copy()
        #a = list(filter(lambda x: np.all(x), pick < 40))
        # TODO: More efficient impl. using map and filter
        if not nogan:
            pick *= 255
        thresh = 30
        for row in range(pick.shape[0]):
            for col in range(pick.shape[1]):
                if np.all(pick[row][col] < thresh):
                    pick[row][col] = [0,0,0]
        pick = Image.fromarray((pick).astype(np.uint8))
        pick = pick.convert("RGBA")
        pixdata = pick.load()

        width, height = pick.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)

        pick.save("testing_"+str(apart)+".png", "PNG")
        pick.load()

        # Paste parts and save pic
        try:
            background.paste(pick, mask=pick.split()[3])  # 3 is the alpha channel
        except IndexError:  # No alpha channel found
            background.paste(pick)

    img_asfloat = (np.array(background)/255).astype(np.float32)
    img_smooth = cv2.medianBlur(img_asfloat, 3) # Median blur only
    #img_smooth = smooth_and_sharp(img_asfloat, alpha=0.2)  # Non-local means
    #img_smooth = img_asfloat   # Keep original
    img = Image.fromarray((img_smooth*255).astype(np.uint8))
    img.save("test_paste.jpg", quality=100)
    print("Saved.")


def create_background():
    background = Image.new("RGB", (64, 64), (0,0,0))
    img = ImageDraw.Draw(background)
    img.ellipse((6, 6, 58, 58), fill=(254,215,50), outline=(220,134,11), width=1)
    #background.save("test.jpg", 'JPEG', quality=100)
    return background


# labels = {"ears": 1, "eyebrows": 2, "eyes": 3, "hands": 4, "mouth": 5, "tears": 6}
# Warning: The ordering of the elements means that the first will be pasted first and the others on top of it
#random.seed(2) # For debugging
combine_parts([1, 5, 3, 2, 4, 6], nogan=True)