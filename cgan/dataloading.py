from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv
from matplotlib.pyplot import imread
from nltk.tokenize import sent_tokenize
import re
import torch
from embeddings.glove_loader import GloveModel

class EmojiCaptionDataset(Dataset):

    def __init__(self, csv_file, embedding_dim, transform=None, use_long_desc=True,
                 glove_path="./embeddings/glove.6B.300d.txt"):
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            csv_rows = [row for row in reader][1:]  # skip header
        if use_long_desc:
            csv_rows_extended = []
            for row in csv_rows:
                id, unicode, img_path, desc_short, desc_long = row
                caption_list = self.__tokenize_desc_long(desc_long)
                for caption in caption_list:
                    csv_rows_extended.append([id, unicode, img_path, desc_short, caption])
            self.data = csv_rows_extended
        else:
            self.data = csv_rows

        self.transform = transform
        self.use_long_desc = use_long_desc

        # Build the GloVe model
        self.glove_model = GloveModel()
        self.glove_model.load(data_dir_path=glove_path, embedding_dim=embedding_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        id, unicode, img_path, desc_short, desc_long = self.data[idx]
        img = imread(img_path)
        if self.use_long_desc:
            caption = desc_long
        else:
            caption = desc_short

        img = self.transform(img)

        caption_embedded = self.glove_model.encode_docs([caption])
        caption_tensor = torch.tensor(caption_embedded)
        return img, caption_tensor

    def __tokenize_desc_long(self, long_desc):
        tokenized = sent_tokenize(long_desc)
        caption_list = []
        for sentence in tokenized:
            filtered_sentence = clean_sentence(sentence)
            caption_list.append(filtered_sentence)
        return caption_list


def clean_sentence(sentence):
    clean_sentence = re.sub(re.compile("[!-/:-@[-`{-~\n]"), "", sentence)
    return clean_sentence


def load_captions_from_textfile(path):
    """
    load captions from a text file were each line contains a caption
    :param path: filepath
    :return: list of captions
    """
    captions = []
    with open(path) as txtfile:
        for line in txtfile:
            caption = clean_sentence(line)  # remove \n
            captions.append(caption)
    return captions