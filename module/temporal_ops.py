import torch
import torch.nn as nn
import ever as er


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)


@er.registry.OP.register()
class TemporalCat(nn.Module):
    def __init__(self, _=None):
        super(TemporalCat, self).__init__()

    def forward(self, features1, features2):
        if isinstance(features1, list):
            change_features = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(features1, features2)]
        else:
            change_features = torch.cat([features1, features2], dim=1)
        return change_features


@er.registry.OP.register()
class TemporalDiff(nn.Module):
    def __init__(self, diff_op):
        super(TemporalDiff, self).__init__()
        self.diff_op = diff_op

    def forward(self, features1, features2):
        if '(x1-x2)**2' == self.diff_op:
            diff_op = lambda x1, x2: (x1 - x2).pow(2)
        elif 'x1x2' == self.diff_op:
            diff_op = lambda x1, x2: x1 * x2
        elif '(x1-x2).abs' == self.diff_op:
            diff_op = lambda x1, x2: (x1 - x2).abs()
        elif 'x1+x2' == self.diff_op:
            diff_op = lambda x1, x2: x1 + x2
        elif 'x1**2+x2**2' == self.diff_op:
            diff_op = lambda x1, x2: x1 ** 2 + x2 ** 2
        else:
            raise ValueError
        if isinstance(features1, list):
            change_features = [diff_op(f1, f2) for f1, f2 in zip(features1, features2)]
        else:
            change_features = diff_op(features1, features2)
        return change_features


class SpatioTemporalInteraction(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, type='conv3d'):
        if type == 'conv3d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, [2, kernel_size, kernel_size], stride=1,
                          padding=(0, kernel_size // 2, kernel_size // 2),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, (2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(True) if kernel_size > 1 else nn.Identity(),
            )


@er.registry.OP.register()
class TemporalSymmetricTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 interaction_type='conv3d',
                 symmetric_fusion='add'):
        super(TemporalSymmetricTransformer, self).__init__()

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, type=interaction_type)

        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(self, features1, features2):
        if isinstance(features1, list):
            d12_features = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                            zip(self.t, features1, features2)]
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features = [self.symmetric_fusion(d12, d21) for d12, d21 in zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        else:
            if self.symmetric_fusion:
                change_features = self.symmetric_fusion(self.t(torch.stack([features1, features2], dim=2)),
                                                        self.t(torch.stack([features2, features1], dim=2)))
            else:
                change_features = self.t(torch.stack([features1, features2], dim=2))
            change_features = change_features.squeeze(dim=2)
        return change_features


if __name__ == '__main__':
    from segmentation_models_pytorch.encoders import get_encoder

    m = get_encoder(name='efficientnet-b0')

    er.param_util.count_model_parameters(TemporalSymmetricTransformer(m.out_channels, m.out_channels, 3, 'conv3d'))
    er.param_util.count_model_parameters(TemporalSymmetricTransformer(m.out_channels, m.out_channels, 3, 'conv1plus2d'))
