import sys
from pathlib import Path
import re
import os
import csv


def preprocess_emoji(dset_path, format='.jpg'):
    """
    :param dset_path: path to emoji image directory
    :return:
    """
    descriptions_dict = {}
    with open('descriptions.csv', 'r') as descriptionsfile:
        csvreader = csv.reader(descriptionsfile, delimiter=',', quotechar='"')
        next(csvreader)  # skip header
        for row in csvreader:
            # unicode value as key, short and long description as value
            descriptions_dict[row[1]] = [row[2],  row[3]]


    # check dataset path
    image_dir = Path(dset_path)
    if image_dir.exists() == False:
        exit('Check your dataset path!')

    # map descriptions to images and save in csv file
    csvheader = ["id", "unicode", "imagepath", "desc_short", "desc_long"]
    csvrows = [csvheader]

    samples_with_caption = 0
    for filename in os.listdir(image_dir):
        if not filename.endswith(format):
            continue
        emoji_unicode = re.split('[_.]', filename)[-2]
        if emoji_unicode in descriptions_dict:
            samples_with_caption += 1
            imagepath = os.path.join(image_dir, filename)
            csv_row = [samples_with_caption, emoji_unicode, imagepath, descriptions_dict[emoji_unicode][0],
                       descriptions_dict[emoji_unicode][1]]
            csvrows.append(csv_row)

    with open("images_descriptions.csv", "w") as img_desc_csv:
        csvwriter = csv.writer(img_desc_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerows(csvrows)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print('Preprocessing emoji dataset...')
        preprocess_emoji(sys.argv[1])
        print('Done!')
    else:
        exit('Check your input arguments!')

