from collections import OrderedDict

from torch import nn

from .layers import (DenseLayer, MultiBranchDenseTDNNBlock, StatsPool,
                     TDNNLayer, TransitLayer)


class DTDNNSS(nn.Module):
    def __init__(self,
                 feat_dim=30,
                 embedding_size=512,
                 num_classes=None,
                 growth_rate=64,
                 bn_size=2,
                 init_channels=128,
                 null=False,
                 reduction=2,
                 config_str='batchnorm-prelu',
                 memory_efficient=True):
        super(DTDNNSS, self).__init__()

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(feat_dim,
                           init_channels,
                           5,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((6, 12), (3, 3), ((1, 3), (1, 3)))):
            block = MultiBranchDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                null=null,
                reduction=reduction,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2
        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))
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
            x = self.classifier(x)
        return x
