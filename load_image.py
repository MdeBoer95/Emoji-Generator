import pandas as pd
import os
from PIL import Image

def load_image(name):
    '''

    :param name: Name of the directory(apple_64, facebook_64 ...)
    :return: list of emoji_image
    '''
    path = os.getcwd()
    path = path + '/' + name + '/'
    name = pd.read_json(path + 'index.json', typ='series')
    image_list = []
    for i in name.values:
        emoji = Image.open(path + i)
        image_list.append(emoji)
    return image_list


if __name__ == "__main__":
    apple_emoji = load_image('apple_64')

