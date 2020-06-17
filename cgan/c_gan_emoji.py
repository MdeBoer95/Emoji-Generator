from __future__ import print_function, division

import torch.nn as nn
from models import Discriminator, Generator
from torch.utils.data import DataLoader
from dataloading import EmojiCaptionDataset, load_captions_from_textfile
import torch
import torch.optim as optim
from torchvision import transforms as transforms, utils as tv_ut
from torch.utils.data import random_split
import os
import argparse
from embeddings.glove_loader import GloveModel


class CGan_Trainer():
    def __init__(self, csv_file="images_descriptions.csv", batch_size=32, device='cpu'):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.embedding_dim = 300
        self.device = device

        self.discriminator = Discriminator(self.img_shape, self.embedding_dim)
        self.generator = Generator(self.latent_dim, self.embedding_dim, self.channels)

        self.batch_size = batch_size
        self.dataset = EmojiCaptionDataset(csv_file, self.embedding_dim, transform=transforms.ToTensor())

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.save_model_path = "./saved_models/"
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)

    def train(self, epochs, save_interval=20):
        # Split data into train and validation set
        train_samples = int(round(len(self.dataset)*0.95))
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
                      % (epoch, epochs, i, len(train_loader),
                         errD.item(), errG.item()))

            if epoch % save_interval == 0 or epoch == epochs-1:
                # Show progress on training set
                self.generate_images(captions, output_name="emojis" + str(epoch) + ".png")
                self.save_model(gen_weights_path=os.path.join(self.save_model_path, "gen_weights" + str(epoch) + ".pt"),
                                dis_weights_path=os.path.join(self.save_model_path, "dis_weights" + str(epoch) + ".pt"))
                # Check performance on validation set
                for i, data in enumerate(val_loader):
                    val_imgs, val_captions = data
                    val_captions = val_captions.to(device)
                    val_imgs = val_imgs.to(device)
                    self.generate_images(val_captions,
                                         output_name="val_emojis_fake" + str(i) + "_" + str(epoch) + ".png")
                    save_image_batch(val_imgs, "val_emojis_real_" + str(i) + "_" + str(epoch) + ".png")

    def load_model(self, gen_weights_path, dis_weights_path):
        self.generator.load_state_dict(torch.load(gen_weights_path))
        self.discriminator.load_state_dict(torch.load(dis_weights_path))

    def save_model(self, gen_weights_path, dis_weights_path):
        torch.save(self.generator.state_dict(), gen_weights_path)
        torch.save(self.discriminator.state_dict(), dis_weights_path)

    def generate_images(self, cond, x_in=None, output_name="generated_emojis.png"):
        if x_in is None:
            x_in = torch.randn(len(cond), 1, self.latent_dim).to(self.device)
        cond = cond.to(self.device)
        self.generator.eval()
        with torch.no_grad():
            output = self.generator(x_in, cond)
            save_image_batch(output.detach(), name=output_name)
        self.generator.train()


def save_image_batch(image_batch, name, output_dir="generated_images"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    grid = tv_ut.make_grid(image_batch, normalize=True, padding=0)
    tv_ut.save_image(grid, os.path.join(output_dir, name))

def train_mode(epochs, batch_size, save_interval, gen_checkpoint, dis_checkpoint):
    # Check if device is available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dcgan = CGan_Trainer(device=device, batch_size=batch_size)
    if gen_checkpoint is not None and dis_checkpoint is not None:
        dcgan.load_model(gen_checkpoint, dis_checkpoint)
    dcgan.train(epochs=epochs, save_interval=save_interval)


def inference_mode(caption_file, gen_weights_path, dis_weights_path, glove_path="./embeddings/glove.6B.300d.txt"):
    # Check if device is available
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dcgan = CGan_Trainer(device=device)
    dcgan.load_model(gen_weights_path, dis_weights_path)
    dcgan.discriminator.to(device)
    dcgan.generator.to(device)
    # Encode captions
    captions = load_captions_from_textfile(caption_file)
    glove_model = GloveModel()
    glove_model.load(data_dir_path=glove_path)
    captions = torch.Tensor([glove_model.encode_docs([c]) for c in captions])
    dcgan.generate_images(captions, output_name="inference_results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        type=str,
                        required=True,
                        help="Train or inference mode")
    parser.add_argument("--epochs",
                        default=5000,
                        type=int,
                        required=False,
                        help="Number of epcohs to train")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        required=False,
                        help="Batch size")
    parser.add_argument("--save_interval",
                        default=100,
                        type=int,
                        required=False,
                        help="Save the model and intermediate results every x epochs")
    parser.add_argument("--gen_checkpoint",
                        default=None,
                        type=str,
                        required=False,
                        help="Load pretrained weights for the generator")
    parser.add_argument("--dis_checkpoint",
                        default=None,
                        type=str,
                        required=False,
                        help="Load pretrained weights for the discriminator")
    parser.add_argument("--inference_caption_file",
                        default=None,
                        type=str,
                        required=False,
                        help="File with captions to perform inference with a trained model. Must be provided if mode is inference")

    args = parser.parse_args()

    if args.mode == 'train':
        train_mode(args.epochs, args.batch_size, args.save_interval, args.gen_checkpoint, args.dis_checkpoint)
    else:
        inference_mode(args.inference_caption_file, args.gen_checkpoint, args.dis_checkpoint)

