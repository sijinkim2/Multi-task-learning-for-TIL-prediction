"""
get_tp_fp_fn, SoftDiceLoss, and DC_and_CE/TopK_loss are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions
"""


from torchvision.transforms import Lambda
from torch import einsum
import numpy as np
from torch.autograd import Variable
from scipy.spatial.distance import cdist
import torch
from torch import nn
from torch.nn import functional as F


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


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


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
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

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", softmax_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", softmax_output) + einsum("bcxyz->bc", y_onehot))
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (
                    einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape  # (batch size,class_num,x,y,z)
        shp_y = gt.shape  # (batch size,1,x,y,z)
        # one hot code for gt
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

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        input = flatten(softmax_output)
        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)


class SSLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        Sensitivity-Specifity loss
        paper: http://www.rogertam.ca/Brosch_MICCAI_2015.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SSLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, net_output, gt, loss_mask=None):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

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

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - softmax_output) ** 2
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1606.04797.pdf
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


class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22

        """
        super(IoULoss, self).__init__()

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

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()

        return -iou


class TverskyLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = 0.3
        self.beta = 0.7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky = tversky[1:]
            else:
                tversky = tversky[:, 1:]
        tversky = tversky.mean()

        return 1 - tversky


class FocalTversky_loss(nn.Module):
    """
    paper: https://arxiv.org/pdf/1810.07842.pdf
    author code: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """

    def __init__(self, tversky_kwargs, gamma=0.75):
        super(FocalTversky_loss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target):
        tversky_loss = 1 + self.tversky(net_output, target)  # = 1-tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, self.gamma)
        return focal_tversky


class AsymLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = 1.5

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)  # shape: (batch size, class num)
        weight = (self.beta ** 2) / (1 + self.beta ** 2)
        asym = (tp + self.smooth) / (tp + weight * fn + (1 - weight) * fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()

        return -asym


def to_one_hot(tensor, nClasses, device):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).to(device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


class mIoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=4, device='cpu'):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        self.device = device

    def forward(self, inputs, target):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        inputs = inputs.to(self.device)
        target = target.to(self.device)

        SMOOTH = 1e-6
        N = inputs.size()[0]

        inputs = F.softmax(inputs, dim=1)
        target_oneHot = to_one_hot(target, self.classes, self.device)
        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2) + SMOOTH

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2) + SMOOTH

        loss = inter / union

        ## Return average loss over classes and batch
        return -loss.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1.0e-8, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]


'''
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

class PenaltyGDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc



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



class ExpLog_loss(nn.Module):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    """
    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3):
        super(ExpLog_loss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, net_output, target):
        dc_loss = -self.dc(net_output, target) # weight=0.8
        wce_loss = self.wce(net_output, target) # weight=0.2
        # with torch.no_grad():
        #     print('dc loss:', dc_loss.cpu().numpy(), 'ce loss:', ce_loss.cpu().numpy())
        #     a = torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma)
        #     b = torch.pow(-torch.log(torch.clamp(ce_loss, 1e-6)), self.gamma)
        #     print('ExpLog dc loss:', a.cpu().numpy(), 'ExpLogce loss:', b.cpu().numpy())
        #     print('*'*20)
        explog_loss = 0.8*torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + \
            0.2*wce_loss

        return explog_loss
'''


class diceloss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()

    def forward(self, pred, target):
        smooth = 1e-5
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class DiceBCELoss(nn.Module):
    """
    naive implementation
    """

    def __init__(self, eps=1e-8, weight=1.0):
        super(DiceBCELoss, self).__init__()
        self._eps = eps
        self._weight = weight

    def forward(self, input, labels):
        preds = F.softmax(input, dim=1)
        intersection = (preds * labels).sum(dim=(2, 3)) + self._eps
        union = (preds.sum(dim=(2, 3)) + labels.sum(dim=(2, 3))) + self._eps
        dice_loss = (1 - (2 * intersection / union)).mean(dim=1)
        dice = (2 * intersection / union)
        bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')
        return self._weight * dice_loss.mean() + bce_loss, dice.mean()


def Class_wise_Dice_score(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01
    pred = F.softmax(input)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels)
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels)
    labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels)

    labels_0 = labels_0.contiguous().view(-1)
    labels_1 = labels_1.contiguous().view(-1)
    labels_2 = labels_2.contiguous().view(-1)

    pred_0 = Lambda(lambda x: x[:, 0, :, :])(pred)
    pred_1 = Lambda(lambda x: x[:, 1, :, :])(pred)
    pred_2 = Lambda(lambda x: x[:, 2, :, :])(pred)

    pred_0 = pred_0.contiguous().view(-1)
    pred_1 = pred_1.contiguous().view(-1)
    pred_2 = pred_2.contiguous().view(-1)

    intersection_0 = (pred_0 * labels_0).sum()
    intersection_1 = (pred_1 * labels_1).sum()
    intersection_2 = (pred_2 * labels_2).sum()
    intersection_SUM = intersection_0 + intersection_1 + intersection_2

    union_0 = (pred_0.sum() + labels_0.sum()) + eps
    union_1 = (pred_1.sum() + labels_1.sum()) + eps
    union_2 = (pred_2.sum() + labels_2.sum()) + eps
    union_SUM = union_0 + union_1 + union_2

    dice_total = (2 * intersection_SUM / union_SUM)
    dice_total_loss = 1 - dice_total

    dice0 = (2 * intersection_0 / union_0)
    dice1 = (2 * intersection_1 / union_1)
    dice2 = (2 * intersection_2 / union_2)
    diceT = dice0 + dice1 + dice2

    dice_loss = (1 - diceT)
    bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')
    return dice0, dice1, dice2, dice_total, bce_loss, dice_total_loss, (bce_loss * b) + dice_total_loss


def Class_wise_Dice_score_multi_task_learning(input, labels, mask_name, batch_size):
    eps = 1e-8
    weight = 1.0
    b = 0.01
    pred = F.softmax(input)

    a = 0
    dice_total_ = 0
    dice_total_loss_ = 0
    bce_loss_ = 0
    intersection_0_ = 0
    intersection_1_ = 0
    intersection_2_ = 0
    union_0_ = 0
    union_1_ = 0
    union_2_ = 0
    dice0_ = 0
    dice1_ = 0
    dice2_ = 0

    for i in range(batch_size):
        # print(mask_name)
        if mask_name[i] == "s":
            a = a + 1

            # print("-----------------------------------")

            labels_ = labels[i]
            labels_unsqueeze = torch.unsqueeze(labels_, dim=0)
            # print(labels_unsqueeze.shape)

            labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels_unsqueeze)
            labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels_unsqueeze)
            labels_2 = Lambda(lambda x: x[:, 2, :, :])(labels_unsqueeze)

            labels_0 = labels_0.contiguous().view(-1)
            labels_1 = labels_1.contiguous().view(-1)
            labels_2 = labels_2.contiguous().view(-1)

            pred_ = pred[i]
            input_ = input[i]
            pred_unsqueeze = torch.unsqueeze(pred_, dim=0)

            pred_0 = Lambda(lambda x: x[:, 0, :, :])(pred_unsqueeze)
            pred_1 = Lambda(lambda x: x[:, 1, :, :])(pred_unsqueeze)
            pred_2 = Lambda(lambda x: x[:, 2, :, :])(pred_unsqueeze)

            pred_0 = pred_0.contiguous().view(-1)
            pred_1 = pred_1.contiguous().view(-1)
            pred_2 = pred_2.contiguous().view(-1)

            intersection_0 = (pred_0 * labels_0).sum()
            intersection_1 = (pred_1 * labels_1).sum()
            intersection_2 = (pred_2 * labels_2).sum()
            intersection_SUM = intersection_0 + intersection_1 + intersection_2

            union_0 = (pred_0.sum() + labels_0.sum()) + eps
            union_1 = (pred_1.sum() + labels_1.sum()) + eps
            union_2 = (pred_2.sum() + labels_2.sum()) + eps
            union_SUM = union_0 + union_1 + union_2

            dice_total = (2 * intersection_SUM / union_SUM)
            dice_total_loss = 1 - dice_total

            dice0 = (2 * intersection_0 / union_0)
            dice1 = (2 * intersection_1 / union_1)
            dice2 = (2 * intersection_2 / union_2)
            # print(dice0, dice1, dice2)
            diceT = dice0 + dice1 + dice2

            # print(dice_total_loss)
            dice_loss = (1 - diceT)
            # N_samples = [pred_0]
            # print(N_samples)
            bce_loss = F.binary_cross_entropy_with_logits(input_, labels_, reduction='mean')
            # print(pred_unsqueeze.shape, labels_unsqueeze.shape)
            # print(bce_loss)

            intersection_0_ = intersection_0_ + intersection_0
            intersection_1_ = intersection_1_ + intersection_1
            intersection_2_ = intersection_2_ + intersection_2
            union_0_ = union_0_ + union_0
            union_1_ = union_1_ + union_1
            union_2_ = union_2_ + union_2
            dice0_ = dice0_ + dice0
            dice1_ = dice1_ + dice1
            dice2_ = dice2_ + dice2
            # print(dice0_, dice1_, dice2_)
            dice_total_ = dice_total_ + dice_total
            dice_total_loss_ = dice_total_loss_ + dice_total_loss
            bce_loss_ = bce_loss_ + bce_loss

    if dice0_ != 0:
        dice0_ = (2 * intersection_0_ / union_0_)
    if dice1_ != 0:
        dice1_ = (2 * intersection_1_ / union_1_)
    if dice2_ != 0:
        dice2_ = (2 * intersection_2_ / union_2_)

    if dice_total_ != 0:
        dice_total_ = dice_total_ / a
    if dice_total_loss_ != 0:
        dice_total_loss_ = dice_total_loss_ / a
    if bce_loss_ != 0:
        bce_loss_ = bce_loss_ / a

    return dice0_, dice1_, dice2_, dice_total_, bce_loss_, dice_total_loss_, (bce_loss_ * b) + dice_total_loss_


def extract_predictions(probabilities, confidence_threshold):
    indices = np.meshgrid(np.arange(0, probabilities.shape[1]), np.arange(0, probabilities.shape[0]))
    indices_x = indices[0]
    indices_y = indices[1]
    indices_x = indices_x.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    indices_y = indices_y.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    probabilities = probabilities.reshape((probabilities.shape[0] * probabilities.shape[1], 1))
    boxes_pred = np.concatenate((indices_x, indices_y, probabilities), axis=1)
    boxes_pred = boxes_pred[np.argsort(boxes_pred[:, 2])[::-1]]
    boxes_pred = boxes_pred[boxes_pred[:, 2] >= confidence_threshold, :]
    return boxes_pred


def non_max_supression_distance(points, distance_threshold):
    log_val = np.ones(points.shape[0])
    wanted = []
    for i in range(points.shape[0]):
        if log_val[i]:
            hit = cdist(np.expand_dims(points[i, :2], 0), points[:, :2])
            hit = np.argwhere(hit <= distance_threshold)
            log_val[hit] = 0
            wanted.append(points[i, :])
    wanted = np.array(wanted)
    return wanted


def count_tp_fp_fn(pred, gt, prob_threshold, hit_distance):
    # This function counts the number of true-positives, false negatives and false positives in a detection taks.

    # inputs: A numpy array of shape [N,3] for pred and gt. N is the number of detections. The first column is the x-position of the detections.
    # the second column is the y-position of the detections. the third column is the probability associated with detections.
    # The hit-distance is the maximum distance of a detection from the ground-truth to be counted as a true-positive, otherwsie, it will be counted as a false positive.
    # The prob_threshld is the threshold value in which detections with probabilities smaller than the threshold value will be discarded.

    # output: number of ture-positives, false positives, and false negatives.
    if pred.shape[0] > 0:
        pred = pred[np.argwhere(pred[:, 2] >= prob_threshold)[:, 0], :2]
        gt = gt[np.argwhere(gt[:, 2] >= prob_threshold)[:, 0], :2]
    if pred.shape[0] > 0:
        if gt.shape[0] > 0:
            hit = cdist(pred[:, :2], gt)
            hit[hit <= hit_distance] = 1
            hit[hit != 1] = 0
            sum_1 = np.sum(hit, 0)
            sum_2 = np.sum(hit, 1)
            fp = sum(sum_2 == 0)
            fn = sum(sum_1 == 0)
            tp = sum(sum_1 != 0)
        else:
            tp = 0
            fn = 0
            fp = pred.shape[0]
    else:
        tp = 0
        fp = 0
        fn = gt.shape[0]
    return tp, fp, fn


######## Calculating TP,FP,FN for each Patch, Add them all and then calculate Sen,FP-Per-Patch

def FROC(pred, gt, confidence_thresholds, hit_distance):
    # this function calculates the average sensitivity and average number of fp_per_image.

    # inputs: pred and gt are lists containing detections from the ground truth and the predictions from the model. for example:
    # pred = [pred_1, pred_2, pred_3], gt = [gt_1, gt_2, gt_3]
    # pred_1 = np.array((N,3))  pred_2 = np.array((M,3))    pred_3 = np.array((P,3))
    # gt_1 = np.array((N,3))    gt_2 = np.array((M,3))      gt_3 = np.array((P,3))
    # confidence_thresholds is the list of the probability thresholds in calculating the FROC curve. for example:
    # confidence_thresholds = np.linspace(0,1,40)
    # hit-distance: is the maximum distance from the ground truth by which a detection will be counted as a true positive.

    # outputs: returns a list for sensitivies and fps_per_image for each of the threshold values.
    sens = []

    fp_img = []
    for threshold in confidence_thresholds:
        tps = []
        fps = []
        fns = []
        for N in range(len(gt)):
            tp, fp, fn = count_tp_fp_fn(pred[N], gt[N], threshold, hit_distance)
            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
        tps = sum(tps)
        fp_per_img = np.mean(fps)
        fps = sum(fps)
        fns = sum(fns)
        if tps != 0:
            sens.append(tps / (tps + fns))
        else:
            sens.append(0)
        fp_img.append(fp_per_img)
    return sens, fp_img


def compute_froc_score(fps_per_image, total_sensitivity, eval_thresholds: tuple = (10, 20, 50, 100, 200, 300)):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the challenge's second evaluation metric, which is defined as the average sensitivity at
    the predefined false positive rates per whole slide image.

    Args:
        fps_per_image: the average number of false positives per image for different thresholds.
        total_sensitivity: sensitivities (true positive rates) for different thresholds.
        eval_thresholds: the false positive rates for calculating the average sensitivity. Defaults
            to (0.25, 0.5, 1, 2, 4, 8) which is the same as the CAMELYON 16 Challenge.

    """

    # total_sensitivity = sum(total_sensitivity)/40
    # fps_per_image = sum(fps_per_image)/40

    # interp_sens = np.interp(eval_thresholds, fps_per_image, total_sensitivity)
    interp_sens = np.interp((10, 20, 50, 100, 200, 300), fps_per_image[::-1], total_sensitivity[::-1])
    return np.mean(interp_sens)


def Class_Wise_TIL_Detection_FROC(input, labels):
    eps = 1e-8
    weight = 1.0
    b = 0.01

    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 18
    confidence_threshold = 0.1
    predicted_detections = []
    ground_truth_detections = []

    preds = F.softmax(input)

    labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels)  # background class of targets
    labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels)  # TIL class of targets
    labels_00 = labels_0.detach().cpu().numpy()
    labels_11 = labels_1.detach().cpu().numpy()

    preds_0 = Lambda(lambda x: x[:, 0, :, :])(preds)  # background class of predictions
    preds_1 = Lambda(lambda x: x[:, 1, :, :])(preds)  # TIL class of predictions
    preds_00 = preds_0.detach().cpu().numpy()
    preds_11 = preds_1.detach().cpu().numpy()

    # get intersection
    intersection_0 = (preds_0 * labels_0).sum()
    intersection_1 = (preds_1 * labels_1).sum()
    intersection_SUM = intersection_0 + intersection_1

    # get union
    union_0 = (preds_0.sum() + labels_0.sum()) + eps
    union_1 = (preds_1.sum() + labels_1.sum()) + eps
    union_SUM = union_0 + union_1

    # get dice total score and dice loss
    dice_total = (2 * intersection_SUM / union_SUM)
    dice_total_loss = 1 - dice_total

    dice0 = (2 * intersection_0 / union_0)
    dice1 = (2 * intersection_1 / union_1)

    bce_loss = F.binary_cross_entropy_with_logits(input, labels, reduction='mean')

    for i in range(len(preds_11)):
        # print(i)
        temp1 = extract_predictions(preds_11[i], confidence_threshold=confidence_threshold)
        predicted_detections.append(non_max_supression_distance(temp1, distance_threshold=18))
    del temp1

    for i in range(len(labels_11)):
        # print(i)
        temp2 = extract_predictions(labels_11[i], confidence_threshold=0.1)
        ground_truth_detections.append(non_max_supression_distance(temp2, distance_threshold=distance_threshold))
    del temp2

    sensitivity, fps_image = FROC(predicted_detections, ground_truth_detections, confidence_thresholds, 8)
    froc_score = compute_froc_score(fps_image, sensitivity)

    print(froc_score)
    # fps_image = int(sum(fps_image)/40)
    # sensitivity = sum(sensitivity)/40

    # sens_loss = 1 - sensitivity

    # print(sensitivity, fps)

    return dice0, dice1, dice_total, bce_loss, dice_total_loss, bce_loss, froc_score


def Class_Wise_TIL_Detection_FROC_multi_task_learning(input, labels, mask_name, batch_size):
    eps = 1e-8
    weight = 1.0
    b = 0.01
    a = 0
    bce_loss_ = 0
    froc_score_ = 0

    confidence_thresholds = np.linspace(0, 1, 40)
    distance_threshold = 18
    confidence_threshold = 0.1
    predicted_detections = []
    ground_truth_detections = []

    pred = F.softmax(input)

    for i in range(batch_size):

        if mask_name[i] == "d":
            a = a + 1
            # print("-----------------------------------")

            labels_ = labels[i]
            labels_unsqueeze = torch.unsqueeze(labels_, dim=0)
            # print(labels_unsqueeze.shape)

            labels_0 = Lambda(lambda x: x[:, 0, :, :])(labels_unsqueeze)
            labels_1 = Lambda(lambda x: x[:, 1, :, :])(labels_unsqueeze)
            labels_00 = labels_0.detach().cpu().numpy()
            labels_11 = labels_1.detach().cpu().numpy()

            pred_ = pred[i]
            input_ = input[i]
            pred_unsqueeze = torch.unsqueeze(pred_, dim=0)

            pred_0 = Lambda(lambda x: x[:, 0, :, :])(pred_unsqueeze)
            pred_1 = Lambda(lambda x: x[:, 1, :, :])(pred_unsqueeze)
            preds_00 = pred_0.detach().cpu().numpy()
            preds_11 = pred_1.detach().cpu().numpy()

            # get intersection
            intersection_0 = (pred_0 * labels_0).sum()
            intersection_1 = (pred_1 * labels_1).sum()
            intersection_SUM = intersection_0 + intersection_1

            # get union
            union_0 = (pred_0.sum() + labels_0.sum()) + eps
            union_1 = (pred_1.sum() + labels_1.sum()) + eps
            union_SUM = union_0 + union_1

            # get dice total score and dice loss
            dice_total = (2 * intersection_SUM / union_SUM)
            dice_total_loss = 1 - dice_total

            dice0 = (2 * intersection_0 / union_0)
            dice1 = (2 * intersection_1 / union_1)

            bce_loss = F.binary_cross_entropy_with_logits(input_, labels_, reduction='mean')

            for i in range(len(preds_11)):
                # print(i)
                temp1 = extract_predictions(preds_11[i], confidence_threshold=confidence_threshold)
                predicted_detections.append(non_max_supression_distance(temp1, distance_threshold=18))
            del temp1

            for i in range(len(labels_11)):
                # print(i)
                temp2 = extract_predictions(labels_11[i], confidence_threshold=0.1)
                ground_truth_detections.append(
                    non_max_supression_distance(temp2, distance_threshold=distance_threshold))
            del temp2

            sensitivity, fps_image = FROC(predicted_detections, ground_truth_detections, confidence_thresholds, 8)
            froc_score = compute_froc_score(fps_image, sensitivity)
            froc_score_ = froc_score_ + froc_score
            bce_loss_ = bce_loss_ + bce_loss

    if froc_score_ != 0:
        froc_score_ = froc_score_ / a
    if bce_loss_ != 0:
        bce_loss_ = bce_loss_ / a

    # return bce_loss, bce_loss, sensitivity, fps_image, froc_score, ground_truth_detections, predicted_detections
    return bce_loss_, froc_score_
