#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.loss_functions.advND_Crossentropy import advCrossentropyND
from nnunet.training.loss_functions.mND_Crossentropy import mCrossentropyND
from nnunet.training.loss_functions.mixND_Crossentropy import mixCrossentropyND
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """

        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

def mget_tp_fp_fn(net_output, gt, focal_conduct, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    # here is a little dummy...for simplicity, focal_conduct is [N,C]
    # y_onehot is [B, C, H, W, D]
    # first reshape [N,C] to [B, H, W, D, C]
    focal_conduct = torch.reshape(focal_conduct, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], net_output.shape[1]))
    # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
    focal_conduct = focal_conduct.transpose(4, 3)
    focal_conduct = focal_conduct.transpose(3, 2)
    focal_conduct = focal_conduct.transpose(2, 1)

    # this is like focal Tversky loss, but it would suppress too much
    # focal_fp = net_output * (1 - y_onehot) * focal_conduct2

    focal_fp = net_output * (1 - y_onehot)
    focal_fn = (1 - net_output) * y_onehot * focal_conduct
    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        focal_fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(focal_fp, dim=1)), dim=1)
        focal_fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(focal_fn, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        focal_fp = focal_fp ** 2
        focal_fn = focal_fn ** 2
        fp = fp ** 2
        fn = fn ** 2

    # print(tp.size())
    tp = sum_tensor(tp, axes, keepdim=False)
    focal_fp = sum_tensor(focal_fp, axes, keepdim=False)
    focal_fn = sum_tensor(focal_fn, axes, keepdim=False)
    # print(tp.size())
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return focal_fp, focal_fn, tp, fp, fn

class mSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        """
        super(mSoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, focal_conduct, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        focal_fp, focal_fn, tp, fp, fn = mget_tp_fp_fn(x, y, focal_conduct, axes, loss_mask, self.square)

        dc = (focal_fp + focal_fn + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return dc

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class advCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs):
        super(advCE_loss, self).__init__()
        self.ce = advCrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss, onehotasy, ydsclossposition = self.ce(net_output, target)
        onehotasy = torch.reshape(onehotasy, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], net_output.shape[1]))
        # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
        onehotasy = onehotasy.transpose(4, 3)
        onehotasy = onehotasy.transpose(3, 2)
        onehotasy = onehotasy.transpose(2, 1)
        ydsclossposition = torch.reshape(ydsclossposition, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], 1))
        ydsclossposition = ydsclossposition.transpose(4, 3)
        ydsclossposition = ydsclossposition.transpose(3, 2)
        ydsclossposition = ydsclossposition.transpose(2, 1)
        ydsclossposition = ydsclossposition.float().cuda()
        dc_loss = self.dc(net_output, onehotasy, loss_mask = ydsclossposition)
        results = ce_loss + dc_loss
        return results

class mCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs):
        super(mCE_loss, self).__init__()
        self.ce = mCrossentropyND(**ce_kwargs)
        self.dc = mSoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss, inpost, focal_conduct = self.ce(net_output, target)
        inpost = torch.reshape(inpost, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], net_output.shape[1]))
        # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
        inpost = inpost.transpose(4, 3)
        inpost = inpost.transpose(3, 2)
        inpost = inpost.transpose(2, 1)
        dc_loss = self.dc(inpost, target, focal_conduct)
        results = ce_loss + dc_loss
        return results

class mixCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs):
        super(mixCE_loss, self).__init__()
        self.ce = mixCrossentropyND(**ce_kwargs)
        self.dc = mSoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, lam):
        ce_loss, y_one_hotmixup, ydsclossposition, inpost, focal_conduct = self.ce(net_output, target, lam)
        y_one_hotmixup = torch.reshape(y_one_hotmixup, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], net_output.shape[1]))
        # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
        y_one_hotmixup = y_one_hotmixup.transpose(4, 3)
        y_one_hotmixup = y_one_hotmixup.transpose(3, 2)
        y_one_hotmixup = y_one_hotmixup.transpose(2, 1)
        ydsclossposition = torch.reshape(ydsclossposition, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], 1))
        ydsclossposition = ydsclossposition.transpose(4, 3)
        ydsclossposition = ydsclossposition.transpose(3, 2)
        ydsclossposition = ydsclossposition.transpose(2, 1)
        ydsclossposition = ydsclossposition.float().cuda()
        inpost = torch.reshape(inpost, (net_output.shape[0], net_output.shape[2], net_output.shape[3], net_output.shape[4], net_output.shape[1]))
        # then use transpose [B, H, W, D, C] to [B, C, H, W, D]
        inpost = inpost.transpose(4, 3)
        inpost = inpost.transpose(3, 2)
        inpost = inpost.transpose(2, 1)
        dc_loss = self.dc(inpost, y_one_hotmixup, focal_conduct, loss_mask = ydsclossposition)
        result = ce_loss + dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result
