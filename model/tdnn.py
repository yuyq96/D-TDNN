from collections import OrderedDict

from torch import nn

from .layers import DenseLayer, StatsPool, TDNNLayer, get_nonlinear


class TDNN(nn.Module):
    def __init__(self,
                 feat_dim=30,
                 embedding_size=512,
                 num_classes=None,
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
        self.nonlinear = get_nonlinear(config_str, embedding_size)
        self.dense = DenseLayer(embedding_size,
                                embedding_size,
                                config_str=config_str)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.xvector(x)
        if self.training:
            x = self.dense(self.nonlinear(x))
            x = self.classifier(x)
        return x
