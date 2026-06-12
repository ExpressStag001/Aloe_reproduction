from itertools import product
import torch as pt
import torch.nn.functional as ptnf


@pt.inference_mode()
def interpolat_argmax_attent(attent, size, mode="bilinear"):
    """Already optimized with PyTorch inference mode.

    - attent: shape=(b,..,s,h,w), dtype=float
    - segment: shape=(b,..,h,w), dtype=int; index segment
    """
    shape0 = attent.shape[:-3]
    attent_ = attent.flatten(0, -4)  # (b*..,s,h,w)
    attent_ = ptnf.interpolate(attent_, size=size, mode=mode)
    segment_ = attent_.argmax(1).byte()  # (b*..,h,w)
    segment = segment_.unflatten(0, shape0)
    return segment


@pt.inference_mode()
def argmax_attent_bg_enhanced(attent, size, mode="bilinear", FG_THRE=0.5):
    """Already optimized with PyTorch inference mode.

    - attent: shape=(b,..,s,h,w), dtype=float
    - segment: shape=(b,..,h,w), dtype=int; index segment
    """
    shape0 = attent.shape[:-3]
    attent_ = attent.flatten(0, -4)  # (b*..,s,h,w)
    attent_ = ptnf.interpolate(attent_, size=size, mode=mode)
    attent_ = bg_enhanced(attent_, FG_THRE)
    segment_ = attent_.argmax(1).byte()  # (b*..,h,w)
    segment = segment_.unflatten(0, shape0)

    return segment


def bg_enhanced(attent, FG_THRE=0.5):
    """
    - attent: shape=(b*..,s,h,w), dtype=float
    """

    attent_ = attent.flatten(-2, -1)  # (b*..,s,h*w)

    # 1. 认定 slot 对 patch 关注度最大值中最小的那一个 slot 为背景 slot
    slz_max = attent_.max(-1)[0]  # (b*..,s)
    bg_idx = slz_max.argmin(-1)  # (b*..)
    bg_slot_mask = pt.zeros_like(slz_max, dtype=pt.bool)  # (b*..,s)
    bg_slot_mask[pt.arange(slz_max.size(0), device=slz_max.device), bg_idx] = True  # (b*..,s)

    # 2. 认定 patch 块中被 slot 关注度最大值中小于阈值的那些 patch 块为背景 patch 块
    pix_max = attent_.max(-2)[0]  # (b*..,h*w)
    bg_pix_mask = (pix_max < FG_THRE)  # (b*..,h*w) bool

    # 3. 将背景 slot 的 attent 中的背景 patch 块的注意力值设置为 1
    attent_[bg_slot_mask.unsqueeze(-1) & bg_pix_mask.unsqueeze(1)] = 1.  # set the background mask score to 1

    return attent_.view_as(attent)

