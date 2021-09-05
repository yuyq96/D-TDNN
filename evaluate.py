import argparse
import os

import numpy as np
import torch
from numpy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import KaldiFeatDataset, Transpose2D
from metric import compute_c_norm, compute_eer, compute_fnr_fpr
from model.dtdnn import DTDNN
from model.dtdnnss import DTDNNSS
from model.tdnn import TDNN

parser = argparse.ArgumentParser(description='Speaker Verification')
parser.add_argument('--root', default='data', type=str)
parser.add_argument('--model',
                    default='D-TDNN',
                    choices=['TDNN', 'D-TDNN', 'D-TDNN-SS'])
parser.add_argument('--checkpoint', default=None, type=str)
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


def evaluate():
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
            utt2emb[utt[0]] = emb

    with open(os.path.join(args.root, 'trials'), 'r') as f:
        scores = []
        labels = []
        for line in f:
            utt1, utt2, label = line.split(' ')
            emb1, emb2 = utt2emb[utt1], utt2emb[utt2]
            score = emb1.dot(emb2) / (linalg.norm(emb1) * linalg.norm(emb2))
            scores.append(score)
            labels.append(1 if label.strip() == 'target' else 0)
        scores = np.array(scores)
        labels = np.array(labels)
        fnr, fpr = compute_fnr_fpr(scores, labels)
        eer, th = compute_eer(fnr, fpr, True, scores)
        print('Equal error rate is {:6f}%, at threshold {:6f}'.format(
            eer * 100, th))
        print('Minimum detection cost (0.01) is {:6f}'.format(
            compute_c_norm(fnr, fpr, 0.01)))
        print('Minimum detection cost (0.001) is {:6f}'.format(
            compute_c_norm(fnr, fpr, 0.001)))


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device(args.device)
    evaluate()
