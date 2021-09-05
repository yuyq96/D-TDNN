import argparse
import os

import kaldiio
import torch
from numpy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import KaldiFeatDataset, Transpose2D
from model.dtdnn import DTDNN
from model.dtdnnss import DTDNNSS
from model.tdnn import TDNN

parser = argparse.ArgumentParser(description='Speaker Verification')
parser.add_argument('--root', default='data', type=str)
parser.add_argument('--model',
                    default='D-TDNN',
                    choices=['TDNN', 'D-TDNN', 'D-TDNN-SS'])
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--norm', default=False, action='store_true')
parser.add_argument('--output', default='vectors', type=str)
parser.add_argument('--device', default="cuda", choices=['cpu', 'cuda'])
parser.add_argument('--pin-memory', default=True, type=bool)


def load_model():
    assert os.path.isfile(
        args.checkpoint), "No checkpoint found at '{}'".format(args.checkpoint)
    print('Loading checkpoint {}'.format(args.checkpoint))
    state_dict = torch.load(args.checkpoint)['state_dict']
    if args.model == 'TDNN':
        model = TDNN()
        del model.nonlinear
        del model.dense
    elif args.model == 'D-TDNN':
        model = DTDNN()
    else:
        model = DTDNNSS()
    model.to(device)
    model.load_state_dict(state_dict)
    return model


def extract():
    model = load_model()
    model.eval()

    transform = Transpose2D()
    dataset = KaldiFeatDataset(root=args.root, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=args.pin_memory)

    utt2emb = {}
    for data, utt in tqdm(loader):
        with torch.no_grad():
            data = data.to(device)
            emb = model(data)[0].cpu().numpy()
            if args.norm:
                emb = emb / linalg.norm(emb)
            utt2emb[utt[0]] = emb
    kaldiio.save_ark(args.output + '.ark', utt2emb, args.output + '.scp')


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device(args.device)
    extract()
