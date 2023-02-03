import torch
import pytorchnet
import argparse
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from distutils.dir_util import copy_tree
from shutil import rmtree, copyfile


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

classes = ['TE', 'NEC', 'LYM', 'TAS']


def main():
    args = parser.parse_args()

    unlabdir = os.path.join(args.data, 'semi_supervised')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    unlab_dataset_format = datasets.ImageFolder(
        unlabdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    unlab_loader_format = torch.utils.data.DataLoader(
        unlab_dataset_format, batch_size=args.batch_size, shuffle=False)

    model = pytorchnet.bagnet9(num_classes=4)

    # create duplicate of dataset
    pardir = os.path.abspath(os.path.join(args.data, os.pardir))
    extdir = os.path.join(pardir, 'extended')
    if os.path.exists(extdir):
        rmtree(extdir)
    copy_tree(args.data, extdir)

    with torch.no_grad():
        for i, (images, _) in enumerate(unlab_loader_format):
            output = model(images)
            pred = torch.argmax(output, dim=1)

            for i in range(len(pred)):
                fname, _ = unlab_loader_format.dataset.samples[i]
                target_path = os.path.join(extdir, 'train', classes[pred[i]],
                                           fname.split('\\')[-1])
                copyfile(fname, target_path)


if __name__ == '__main__':
    main()
