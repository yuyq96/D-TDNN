from collections import OrderedDict

from torch import nn

from .layers import TDNNLayer, DenseLayer, StatsPool


class TDNN(nn.Module):
    def __init__(self, feat_dim=30, embedding_size=512,
                 config_str='batchnorm-relu'):
        super(TDNN, self).__init__()

        self.xvector = nn.Sequential(OrderedDict([
            ('tdnn1', TDNNLayer(feat_dim, 512, 5, dilation=1, padding=-1,
                                config_str=config_str)),
            ('tdnn2', TDNNLayer(512, 512, 3, dilation=2, padding=-1,
                                config_str=config_str)),
            ('tdnn3', TDNNLayer(512, 512, 3, dilation=3, padding=-1,
                                config_str=config_str)),
            ('tdnn4', DenseLayer(512, 512, config_str=config_str)),
            ('tdnn5', DenseLayer(512, 1500, config_str=config_str)),
            ('stats', StatsPool()),
            ('affine', nn.Linear(3000, embedding_size))
        ]))

    def forward(self, x):
        return self.xvector(x)
