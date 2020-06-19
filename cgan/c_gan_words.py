from __future__ import print_function, division

from cgan.dataloading import EmojiCaptionDataset
import torch
import argparse
from cgan.trainer import CGanTrainer

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

    # Check if device is available and load checkpoints
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    embedding_dim = 300
    dataset = EmojiCaptionDataset("images_descriptions.csv", embedding_dim)
    cgan = CGanTrainer(dataset=dataset, embedding_dim=embedding_dim, device=device, batch_size=args.batch_size)
    if args.gen_checkpoint and args.dis_checkpoint:
        cgan.load_model(args.gen_checkpoint, args.dis_checkpoint)
    # Run training or inference
    if args.mode == 'train':
        cgan.train(epochs=args.epochs, save_interval=args.save_interval)
    elif args.mode == 'inference':
        cgan.inference(args.inference_caption_file)
    else:
        raise ValueError("--mode must be one of {train, inference}")

