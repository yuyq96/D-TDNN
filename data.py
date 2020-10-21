import os
import random

import librosa
import numpy as np
from kaldifeat import compute_mfcc_feats, compute_vad, apply_cmvn_sliding
from torch.utils.data import Dataset


class KaldiFeatDataset(Dataset):

    def __init__(self, root, transform=None):
        self.transform = transform
        self.egs = []
        with open(os.path.join(root, 'wav.scp'), 'r') as f:
            for line in f:
                utt, path = line.strip().split()
                self.egs.append([path, utt])

    def __len__(self):
        return len(self.egs)

    @staticmethod
    def mfcc(waveform):
        waveform = (waveform * 32768).astype(np.int16)
        return compute_mfcc_feats(
            waveform=waveform,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
            num_ceps=30,
            num_mel_bins=30,
            low_freq=20,
            high_freq=7600,
            snip_edges=False,
            dtype=np.float32
        )

    @staticmethod
    def vad(feats):
        return compute_vad(
            log_energy=feats[:, 0],
            energy_mean_scale=0.5,
            energy_threshold=5.5,
            frames_context=2,
            proportion_threshold=0.12
        )

    @staticmethod
    def cmn(feats):
        return apply_cmvn_sliding(
            feat=feats,
            center=True,
            window=300,
            min_window=100,
            norm_vars=False
        )

    def __getitem__(self, item):
        path, utt = self.egs[item]
        y, _ = librosa.load(path, 16000)
        feats = self.mfcc(y)
        frames = self.vad(feats)
        feats = self.cmn(feats)
        feats = feats[frames]
        if self.transform is not None:
            feats = self.transform(feats)
        return feats, utt


class Transpose2D(object):

    def __call__(self, a):
        return a.transpose((1, 0))
