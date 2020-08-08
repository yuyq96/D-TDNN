import os

import kaldiio
from torch.utils.data import Dataset


class KaldiFeatDataset(Dataset):

    def __init__(self, root, transform=None):
        super(KaldiFeatDataset, self).__init__()
        self.transform = transform
        self.feats = []
        with open(os.path.join(root, 'feats.scp'), 'r') as f:
            for line in f:
                utt, feats = line.split(' ')
                self.feats.append((feats, utt))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        feats, utt = self.feats[index]
        feats = kaldiio.load_mat(feats)
        if self.transform is not None:
            feats = self.transform(feats)
        return feats, utt


class Transpose2D(object):

    def __call__(self, a):
        return a.transpose((1, 0))
