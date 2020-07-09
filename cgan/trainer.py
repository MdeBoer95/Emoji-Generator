from __future__ import print_function, division

import datetime

import torch.nn as nn

from cgan.models import Discriminator, Generator
from torch.utils.data import DataLoader
from cgan.dataloading import load_captions_from_textfile
import torch
import torch.optim as optim
from torchvision import utils as tv_ut
from torch.utils.data import random_split
import os
import numpy as np
from cgan.embeddings.glove_loader import GloveModel
from cgan.combiner import combine_parts, sort_parts


class CGanTrainer():
    def __init__(self, dataset, embedding_dim, batch_size=32, device='cpu'):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.embedding_dim = embedding_dim
        self.device = device

        self.discriminator = Discriminator(self.img_shape, self.embedding_dim)
        self.generator = Generator(self.latent_dim, self.embedding_dim, self.channels)

        self.batch_size = batch_size
        self.dataset = dataset

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.save_model_path = "./saved_models/"
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)

        self.val_output_dir = "./val_output"
        self.train_output_dir = "./train_output"

    def train(self, epochs, save_interval=20):
        # Split data into train and validation set
        train_samples = int(round(len(self.dataset)*0.90))
        train_set, val_set = random_split(self.dataset, [train_samples, len(self.dataset) - train_samples])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        d_net = self.discriminator
        g_net = self.generator
        device = self.device
        d_net.to(device)
        g_net.to(device)

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                imgs, captions = data

                b_size = imgs.size(0)
                # create valid and fake labels
                valid_labels = torch.ones(b_size).to(device)
                fake_labels = torch.zeros(b_size).to(device)

                captions = captions.to(device)
                imgs = imgs.to(device)

                # Train with all real batch
                d_net.zero_grad()
                output = d_net(imgs, captions)
                errD_real = self.criterion(output.squeeze(), valid_labels)
                errD_real.backward()

                # Train with all-fake batch
                input_noise = torch.randn(b_size, 1, self.latent_dim).to(device)
                fake_imgs = g_net(input_noise, captions)
                output = d_net(fake_imgs, captions)
                errD_fake = self.criterion(output.squeeze(), fake_labels)
                errD_fake.backward(retain_graph=True)
                errD = errD_real + errD_fake
                self.optimizer_d.step()

                # Update Generator
                g_net.zero_grad()
                output = d_net(fake_imgs, captions).view(-1)
                errG = self.criterion(output.squeeze(), valid_labels)
                errG.backward()
                self.optimizer_g.step()

                # Output training stats
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                      % (epoch, epochs, i, len(train_loader) - 1,
                         errD.item(), errG.item()))

            if epoch % save_interval == 0 or epoch == epochs-1:
                # Show progress on training set
                self.generate_images(captions,
                                     output_path=os.path.join(self.train_output_dir, "emojis" + str(epoch) + ".png"))
                self.save_model(gen_weights_path=os.path.join(self.save_model_path, "gen_weights" + str(epoch) + ".pt"),
                                dis_weights_path=os.path.join(self.save_model_path, "dis_weights" + str(epoch) + ".pt"))

                # Check performance on validation set
                for i, data in enumerate(val_loader):
                    val_imgs, val_captions = data
                    val_captions = val_captions.to(device)
                    val_imgs = val_imgs.to(device)
                    self.generate_images(val_captions, output_path=os.path.join(self.val_output_dir, "val_emojis_fake" + str(epoch) + "_" + str(i) + ".png"))
                    # Save real images from validation set for comparison
                    real_imgs_path = os.path.join(self.val_output_dir, "val_emojis_real" + str(i) + "_.png")
                    if not os.path.exists(real_imgs_path):
                        save_image_batch(val_imgs, output_path=real_imgs_path)

    def load_model(self, gen_weights_path, dis_weights_path, map_location=None):
        self.generator.load_state_dict(torch.load(gen_weights_path, map_location=map_location))
        self.discriminator.load_state_dict(torch.load(dis_weights_path, map_location=map_location))

    def save_model(self, gen_weights_path, dis_weights_path):
        torch.save(self.generator.state_dict(), gen_weights_path)
        torch.save(self.discriminator.state_dict(), dis_weights_path)

    def generate_images(self, cond, output_path, x_in=None, grid=True, names=["ears/", "eyebrows/", "eyes/", "hands/", "mouth/", "tears/"]):
        if x_in is None:
            x_in = torch.randn(len(cond), 1, self.latent_dim).to(self.device)
        cond = cond.to(self.device)
        self.generator.eval()
        with torch.no_grad():
            output = self.generator(x_in, cond)
            save_image_batch(output.detach(), output_path, grid=grid, names=names)
        self.generator.train()

    def inference(self, captions, output_path="inference_results/", glove_model=None, x_in=None, mode='word'):
        self.discriminator.to(self.device)
        self.generator.to(self.device)
        if type(captions) is list:
            captions = captions
        else:
            captions = load_captions_from_textfile(captions)

        if mode == 'word':
            if not glove_model:
                glove_model = GloveModel()
                glove_model.load("cgan/embeddings/glove.6B.300d.txt")
            captions = torch.Tensor([glove_model.encode_docs([c]) for c in captions])
            grid = True
            self.generate_images(captions, output_path=output_path, x_in=x_in, grid=grid)

        elif mode == 'segment':
            captions = sort_parts(captions)
            captions_names = [{1: "ears/", 2: "eyebrows/", 3: "eyes/", 4: "hands/", 5: "mouth/", 6: "tears/"}[capt] for capt in captions]
            onehots = []
            for label in captions:
                onehot = np.zeros((1, self.embedding_dim))   # Dimension of one-hot vectors should be always num_classes
                onehot[0, label - 1] = 1
                onehots.append(onehot)
            oh_captions = torch.Tensor(onehots)
            grid = False
            self.generate_images(oh_captions, output_path="inference_results/", x_in=x_in, grid=grid, names=captions_names)
            combine_parts(captions, nogan=False, output_path=output_path)
        else:
            raise ValueError("mode must be one of {word, segments}")


def save_image_batch(image_batch, output_path, grid=True, names=["ears/", "eyebrows/", "eyes/", "hands/", "mouth/", "tears/"]):
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if grid:
        grid = tv_ut.make_grid(image_batch, normalize=True, padding=0)
        tv_ut.save_image(grid, output_path)
    else:
        for i in range(image_batch.size(0)):
            name = names[i]
            name_dir = os.path.dirname(output_path+name)
            if name_dir != '' and not os.path.exists(name_dir):
                os.mkdir(name_dir)
            tv_ut.save_image(image_batch[i, :, :, :], output_path+name+'{}.png'.format(datetime.datetime.now().strftime("%f")))


def five():
    return 5

def five_string():
    return "Five"

