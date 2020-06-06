import pandas as pd
import os
from PIL import Image
import json

EMOJI_DIRS = ["apple_64", "facebook_64", "google_64", "twitter_64", "wechat_64"]

def load_image(name):
    '''

    :param name: Name of the directory(apple_64, facebook_64 ...)
    :return: list of emoji_image
    '''
    path = os.getcwd()
    path = path + '/' + name + '/'
    name = pd.read_json(path + 'index.json', typ='series')
    image_dict = {}
    for name in name.values:
        emoji = Image.open(path + name)
        image_dict[name] = emoji
    return image_dict

def move_to_one_folder(output_dir, emoji_dirs):
    """
    Save all emoji images in one directory (needed for stylegan)
    :param emoji_dirs: list of names of the emoji-directories to combine
    :param output_dir
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    index_json = {}
    img_count = 0
    for dir in emoji_dirs:
        emojis = load_image(dir)
        for name, img in emojis.items():
            new_img_name = dir + "_" + name
            img.save(os.path.join(output_dir, new_img_name), "PNG")
            index_json[img_count] = new_img_name
            img_count += 1

    with open(os.path.join(output_dir, 'index.json'), 'w') as fp:
        json.dump(index_json, fp)


if __name__ == "__main__":
    apple_emoji = load_image('apple_64')
    move_to_one_folder("all_emojis", EMOJI_DIRS[0:4]) # currently not using wechat emojis since they are much smaller

