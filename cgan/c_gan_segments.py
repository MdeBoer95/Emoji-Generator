import argparse
import torch
from cgan.trainer import CGanTrainer
from cgan.dataloading import EmojiFaceSegments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default='inference',
                        type=str,
                        required=False,
                        help="Train or inference mode") # TODO implement inference
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
                        default=2,
                        type=int,
                        required=False,
                        help="Save the model and intermediate results every x epochs")
    parser.add_argument("--gen_checkpoint",
                        default="saved_models/gen_weights516.pt",
                        type=str,
                        required=False,
                        help="Load pretrained weights for the generator")
    parser.add_argument("--dis_checkpoint",
                        default="saved_models/dis_weights516.pt",
                        type=str,
                        required=False,
                        help="Load pretrained weights for the discriminator")

    args = parser.parse_args()

    # Check if device is available and load checkpoints
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dataset_path = "../segmentation/segmentation_data"
    dataset = EmojiFaceSegments(dataset_path)
    embedding_dim = len(dataset.labels)
    cgan = CGanTrainer(dataset=dataset, embedding_dim=embedding_dim, device=device, batch_size=args.batch_size)
    if args.gen_checkpoint and args.dis_checkpoint:
        cgan.load_model(args.gen_checkpoint, args.dis_checkpoint)
    # Run training or inference
    if args.mode == 'train':
        cgan.train(epochs=args.epochs, save_interval=args.save_interval)
    elif args.mode == 'inference':
        cgan.inference([1, 2, 3, 4, 5, 6], mode='segment')  # Create an image for each class
    else:
        raise ValueError("--mode must be one of {train, inference}")

