import ever as er
import ever.module as erm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder

from core import field

MAX_TIMES = 50


def generate_target(x1, y, pe=True):
    # x [N, C * 0, H, W]
    # y dict(mask1=tensor[N, H, W], ...)
    mask1 = y[field.MASK1]
    N = x1.size(0)
    org_inds = np.arange(N)
    t = 0
    while True and t <= MAX_TIMES:
        t += 1
        shuffle_inds = org_inds.copy()
        np.random.shuffle(shuffle_inds)

        ok = org_inds == shuffle_inds
        if all(~ok):
            break
    if pe:
        virtual_x2 = x1[shuffle_inds, :, :, :]
        virtual_mask2 = mask1[shuffle_inds, ...]
        x = torch.cat([x1, virtual_x2], dim=1)
        y[field.VMASK2] = virtual_mask2
    else:
        x = torch.cat([x1, y[field.COLOR_TRANS_X]], dim=1)
        y[field.VMASK2] = mask1
    return x, y, shuffle_inds


def get_decoder(name):
    if 'Unet' == name:
        return UnetDecoder
    if 'FPN' == name:
        return FPNDecoder

    raise ModuleNotFoundError


class ListInputWrapper(nn.Module):
    def __init__(self, module):
        super(ListInputWrapper, self).__init__()
        self.m = module

    def forward(self, inputs):
        return self.m(*inputs)


class SegmentationLoss(object):
    def semantic_loss(self, y_true: torch.Tensor, y_pred, loss_config, prefix='', weight_cls=None):
        loss_dict = dict()

        if 'ce' in loss_config:
            if 'weight' in loss_config.ce:
                weight = loss_config.ce.weight
            else:
                weight = 1.0

            loss_dict[f'{prefix}ce_loss'] = weight * F.cross_entropy(y_pred, y_true.long(), weight=weight_cls,
                                                                     ignore_index=loss_config.ignore_index)

        return loss_dict

    def change_loss(self, y_true, y_pred, loss_config):
        if 'change_weight' in loss_config:
            weight = loss_config.change_weight
        else:
            weight = 1.0
        if 'tver' in loss_config:
            alpha = loss_config.tver.alpha
            beta = loss_config.tver.beta
            loss_dict = dict(
                tver_loss=weight * erm.loss.tversky_loss_with_logits(y_pred, y_true, alpha=alpha, beta=beta,
                                                                     ignore_index=loss_config.ignore_index)
            )
        else:
            loss_dict = dict(
                dice_loss=weight * erm.loss.dice_loss_with_logits(y_pred, y_true,
                                                                  ignore_index=loss_config.ignore_index),
            )

        loss_dict['bce_loss'] = weight * erm.loss.binary_cross_entropy_with_logits(y_pred, y_true,
                                                                                   ignore_index=loss_config.ignore_index)

        return loss_dict

    def uda_loss(self, pred_target, pred_source, loss_config):
        """
            Entropy loss for probabilistic prediction vectors
            input: batch_size x channels x h x w
            output: batch_size x 0 x h x w
        """
        loss_dict = dict()
        n, c, h, w = pred_target.size()
        if 'uda' in loss_config:
            pred_target = F.softmax(pred_target, dim=1)
            pred_source = F.softmax(pred_source, dim=1)
            if "minent" in loss_config.uda:
                if 'weight' in loss_config.uda.minent:
                    weight = loss_config.uda.minent.weight

                else:
                    weight = 1.0
                loss_dict['minent_loss'] = weight * (
                        -torch.sum(torch.mul(pred_target, torch.log2(pred_target + 1e-30))) / (n * h * w * np.log2(c)))

            if "minent_cross" in loss_config.uda:
                if 'weight' in loss_config.uda.minent_cross:
                    weight = loss_config.uda.minent_cross.weight
                else:
                    weight = 1.0

                classes = self.config.semantic_decoder.classifier.out_channels
                classes_type = list(range(0, classes ** 2, 1))
                classes_unchange = [i + classes * i for i in range(classes)]
                classes_change = list(set(classes_type) - set(classes_unchange))
                pred_source = pred_source[:, classes_unchange, :, :]
                pred_target = pred_target[:, classes_change, :, :]
                ent_source = torch.mul(pred_source, torch.log2(pred_source + 1e-30))
                ent_target = torch.mul(pred_target, torch.log2(pred_target + 1e-30))
                if loss_config.uda.minent_cross.only_change:
                    loss_dict['minent_cross_loss'] = weight * (-torch.sum(ent_target)) / (n * h * w * np.log2(c))
                else:
                    loss_dict['minent_cross_loss'] = weight * (-torch.sum(ent_target) - torch.sum(ent_source)) / (
                                n * h * w * np.log2(c))

            return loss_dict

class ChangeMask(nn.Module):
    def __init__(self,
                 encoder,
                 temporal_transformer,
                 semantic_decoder,
                 change_decoder,
                 ):
        super(ChangeMask, self).__init__()

        self.encoder = encoder
        self.temporal_transformer = temporal_transformer
        self.semantic_decoder = semantic_decoder
        self.change_decoder = change_decoder

    def forward(self, x):
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]

        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        temporal_features = self.temporal_transformer(features1, features2)
        semantic_logit1 = self.semantic_decoder(features1)
        semantic_logit2 = self.semantic_decoder(features2)
        change_logit = self.change_decoder(temporal_features)
        temporal_features_T = self.temporal_transformer(features2, features1)
        change_logit_T = self.change_decoder(temporal_features_T)

        # return change_logit
        # return torch.cat([semantic_logit1, semantic_logit2, change_logit], dim=0)
        return torch.cat([semantic_logit1, semantic_logit2, change_logit, change_logit_T], dim=1)


@er.registry.MODEL.register()
class UPchange(er.ERModule, SegmentationLoss):
    def __init__(self, config):
        super(UPchange, self).__init__(config)
        encoder = get_encoder(name=self.config.encoder.name, weights=self.config.encoder.weights)
        if self.config.temporal_transformer.type == 'TemporalSymmetricTransformer':
            temporal_transformer = er.registry.OP[self.config.temporal_transformer.type](
                in_channels=encoder.out_channels,
                out_channels=encoder.out_channels,
                **self.config.temporal_transformer.params)
        else:
            temporal_transformer = er.registry.OP[self.config.temporal_transformer.type](
                **self.config.temporal_transformer.params)

        semantic_decoder = nn.Sequential(
            ListInputWrapper(get_decoder(self.config.semantic_decoder.name)(
                encoder_channels=encoder.out_channels,
                **self.config.semantic_decoder.decoder
            )),
            erm.ConvUpsampling(**self.config.semantic_decoder.classifier)
        )
        if self.config.temporal_transformer.type == 'TemporalCat':
            outcs = [2 * outc for outc in encoder.out_channels]
        elif self.config.temporal_transformer.type == 'TemporalDiff':
            outcs = encoder.out_channels
        elif self.config.temporal_transformer.type == 'TemporalSymmetricTransformer':
            outcs = encoder.out_channels

        change_decoder = nn.Sequential(
            ListInputWrapper(get_decoder(self.config.change_decoder.name)(
                encoder_channels=outcs,
                **self.config.change_decoder.decoder
            )),
            erm.ConvUpsampling(**self.config.change_decoder.classifier)
        )

        self.changemask = ChangeMask(
            encoder=encoder,
            temporal_transformer=temporal_transformer,
            semantic_decoder=semantic_decoder,
            change_decoder=change_decoder
        )
        self.postprocess_flag = True
        self.idx = 0
        num_classes = self.config.change_decoder.classifier.out_channels
        self.change_stat = np.array([0 for i in range(num_classes)])
        self.change_stat_pesudo = np.array([0 for i in range(num_classes)])

        # self.interp_target = nn.Upsample(size=(input_size_target[0], input_size_target[0]), mode='bilinear',
        #                         align_corners=True)

    def forward(self, img_target, x=None, y=None, postprocess=True, weight=None, source=True):
        if self.training:
            if x.size(1) == 3:
                if self.config.use_constract:
                    self.idx += 1
                    if self.idx % 10 == 0:
                        x, y, _ = generate_target(x, y, pe=False)
                    else:
                        x, y, _ = generate_target(x, y)
                else:
                    x, y, _ = generate_target(x, y)

            cat_logits = self.changemask(x)
        else:
            cat_logits = self.changemask(img_target)

        if self.training:
            classes = self.config.semantic_decoder.classifier.out_channels
            logit1 = cat_logits[:, :classes, :, :]
            logit2 = cat_logits[:, classes:classes * 2, :, :]
            change_logit = cat_logits[:, classes * 2:classes * 2 + classes ** 2, :, :]
            change_logit_T = cat_logits[:, classes * 2 + classes ** 2:, :, :]

            try:
                y1_true = y[field.MASK1]
                vy2_true = y[field.VMASK2]
            except:
                y1_true = y[0]
                vy2_true = y[1]

            change_mask = y1_true * classes + vy2_true
            change_mask = torch.where((y1_true == -1) | (vy2_true == -1),
                                      torch.tensor(-1, device=y1_true.device, dtype=torch.int32),
                                      change_mask)

            change_stat = change_mask.cpu().numpy()
            change_stat = change_stat.ravel()
            valid_inds = np.where(change_stat != -1)
            change_stat = change_stat[valid_inds]
            change_stat = np.bincount(change_stat, minlength=classes ** 2)
            if source:
                self.change_stat += change_stat
            else:
                self.change_stat_pesudo += change_stat

            change_mask_T = vy2_true * classes + y1_true
            change_mask_T = torch.where((y1_true == -1) | (vy2_true == -1),
                                        torch.tensor(-1, device=y1_true.device, dtype=torch.int32),
                                        change_mask_T)

            binary_mask = torch.where(y1_true != vy2_true,
                                      torch.tensor(0, device=y1_true.device, dtype=torch.int32),
                                      torch.tensor(1, device=y1_true.device, dtype=torch.int32), )
            binary_mask = torch.where((y1_true == -1) | (vy2_true == -1),
                                      torch.tensor(0, device=y1_true.device, dtype=torch.int32),
                                      binary_mask)

            loss_dict = dict()
            if weight is not None:
                loss_dict.update(self.semantic_loss(change_mask, change_logit, self.config.loss, weight_cls=weight))
                if self.config.use_trans:
                    loss_dict.update(
                        self.semantic_loss(change_mask_T, change_logit_T, self.config.loss, "T", weight_cls=weight))
            else:
                loss_dict.update(self.semantic_loss(change_mask, change_logit, self.config.loss))
                if self.config.use_trans:
                    loss_dict.update(self.semantic_loss(change_mask_T, change_logit_T, self.config.loss, "T"))

            if self.config.use_semantic:
                loss_dict.update(self.semantic_loss(y1_true, logit1, self.config.loss, 't1'))
                loss_dict.update(self.semantic_loss(vy2_true, logit2, self.config.loss, 't2'))

            if "uda" in self.config.loss and img_target is not None:
                pred_targets = self.changemask(img_target)
                loss_dict.update(self.uda_loss(pred_targets[:, classes * 2: classes * 2 + classes ** 2, :, :],
                                               cat_logits[:, classes * 2: classes * 2 + classes ** 2, :, :],
                                               self.config.loss))

                if self.config.use_feature:
                    return loss_dict, pred_targets[:, classes * 2: classes * 2 + classes ** 2, :, :], cat_logits[:,
                                                                                                      classes * 2: classes * 2 + classes ** 2,
                                                                                                      :, :]
                else:
                    return loss_dict
            else:
                return loss_dict

        if postprocess and self.postprocess_flag:
            return self.postprocess(cat_logits)
        else:
            return cat_logits

    def use_postprocess(self, flag):
        self.postprocess_flag = flag

    def postprocess(self, cat_logits):
        # cat_logits [N, 2C+0, H, W]
        classes = self.config.semantic_decoder.classifier.out_channels
        cat_logits = cat_logits.detach()

        pred = cat_logits[:, classes * 2: classes * 2 + classes ** 2, :, :].argmax(dim=1).to(dtype=torch.int32)
        pred_softmax = torch.softmax(cat_logits[:, classes * 2: classes * 2 + classes ** 2, :, :], dim=1)
        pred_ent = -torch.sum(torch.mul(pred_softmax, torch.log2(pred_softmax + 1e-30)), dim=1)
        prob = torch.amax(pred_softmax, dim=1)
        # pred = cat_logits.argmax(dim=0).to(dtype=torch.int32)
        pred1 = (pred // classes).to(dtype=torch.int32)
        pred2 = pred % classes

        change_pred = torch.where(pred1 == pred2, torch.zeros_like(pred1), torch.ones_like(pred1))
        #change_pred = torch.where(pred_ent>2.5,torch.zeros_like(pred1),change_pred)

        pred1 = cat_logits[:, :classes, :, :].argmax(dim=1).to(dtype=torch.int32)
        pred2 = cat_logits[:, classes:classes * 2, :, :].argmax(dim=1).to(dtype=torch.int32)

        cat_logits = torch.cat((pred1, pred2, change_pred, pred_ent), dim=0)

        return (torch.where(change_pred == 1, pred1 * classes + pred2 + 1, torch.zeros_like(pred)),
                cat_logits)

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(name='', weights=None),
            shared_encoder=True,
            temporal_transformer=dict(
                type='TemporalCat',
                params=dict(
                )
            ),
            semantic_decoder=dict(),
            change_decoder=dict(),
            loss=dict(
                ignore_index=-1,
                ce=dict(),
                tver=dict(alpha=0.5, beta=0.5)
            ),
        ))

    def log_info(self):
        return dict(
            backbone=self.config.encoder.name,
            temporal_transformer=self.config.temporal_transformer.type,
            semantic_decoder=self.config.semantic_decoder.name,
            change_decoder=self.config.change_decoder.name
        )


if __name__ == '__main__':
    cfg = er.config.import_config(r'F:\git_projects\ever\projects\ChangeMask\configs\hiucdv1\changemask\ef_tst_u.py')
    m = ChangeMaskVariants(cfg.model.params)

    er.param_util.count_model_params_flops(m, torch.ones(1, 6, 256, 256))
