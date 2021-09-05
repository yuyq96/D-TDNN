import os

import kaldiio
from torch.utils.data import Dataset


class KaldiFeatDataset(Dataset):
    def __init__(self, root, transform=None, label='utt'):
        super(KaldiFeatDataset, self).__init__()
        self.transform = transform
        self.feats = []
        self.utt2sid = None
        spk2sid = {}
        cnt = 0
        if label == 'sid':
            self.utt2sid = {}
            with open(os.path.join(root, 'utt2spk'), 'r') as f:
                for line in f:
                    utt, spk = line.split()
                    if spk2sid.get(spk) is None:
                        spk2sid[spk] = cnt
                        cnt += 1
                    self.utt2sid[utt] = spk2sid[spk]
        with open(os.path.join(root, 'feats.scp'), 'r') as f:
            for line in f:
                utt, feats = line.split()
                self.feats.append((feats, utt))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        feats, utt = self.feats[index]
        feats = kaldiio.load_mat(feats)
        if self.transform is not None:
            feats = self.transform(feats)
        if self.utt2sid is not None:
            sid = self.utt2sid[utt]
            return feats, sid
        return feats, utt


class Transpose2D(object):
    def __call__(self, a):
        return a.transpose((1, 0))
