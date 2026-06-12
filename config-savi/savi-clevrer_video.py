from einops import rearrange
import torch.nn.functional as ptnf
import math

# 为对齐 slotformer , 当前代码在评估细节做了如下修改：
# 1. _acc_dict_bgeh 使用 " b t h w s -> (b t) (h w) s " 而非 ” b t h w s -> b (t h w) s “
# 2. miou skip=[0]

from object_centric_bench.datum import (
    Resize,
    Normalize,
    Lambda,
    CLEVRER_Video,
    ClPadToMax1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    GaussianVarianceKLDLoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveCkpt,
    ExtractSlots,
)
from object_centric_bench.model import (
    SAVi,
    CNN,
    MLP,
    SlotformerLatentsInit,
    GaussianProjection,
    SlotAttention_rep,
    Linear,
    LayerNorm,
    CartesianPositionalEmbedding2d,
    BroadcastCNNDecoder,
    ResidualMLPTransit,
)
from object_centric_bench.util import Compose, ComposeNoStar, move_to_cuda
from object_centric_bench.util_model import interpolat_argmax_attent, argmax_attent_bg_enhanced

### global
slot_num = 7
resolut0 = [64, 64]
broadcast_resolut = [8, 8]  # slot 空间广播输出的尺寸
slot_dim = 128
vfm_dim = 128

total_step = 230000
val_interval = total_step // 40
batch_size_t = 64
batch_size_v = batch_size_t
num_work = 8
lr = 1e-4

video_len = 128
sample_frames = 6  # 稀疏采样视频后的帧数
sample_offset = 1  # 稀疏采样间隔
clip_len = 8  # for val/extracct

### datum
IMAGENET_MEAN = [[[127.5]], [[127.5]], [[127.5]]]
IMAGENET_STD  = [[[127.5]], [[127.5]], [[127.5]]]

transform_t = [
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = transform_t
dataset_t = dict(
    type=CLEVRER_Video,
    data_file="clevrer/train.lmdb",
    base_dir=...,
    extra_keys=["question","segment"],
    transform=dict(type=Compose, transforms=transform_t),
    video_len = video_len,
    sample_frames=sample_frames,
    sample_offset=sample_offset,
    sample_mode="train",
)
dataset_v = dict(
    type=CLEVRER_Video,
    data_file="clevrer/val.lmdb",
    base_dir=...,
    extra_keys=["question","segment"],
    transform=dict(type=Compose, transforms=transform_v),
    video_len = video_len,
    sample_frames=sample_frames,
    sample_offset=sample_offset,
    sample_mode="val",
)
dataset_test = dict(
    type=CLEVRER_Video,
    data_file="clevrer/test.lmdb",
    base_dir=...,
    extra_keys=["question","segment"],
    transform=dict(type=Compose, transforms=transform_v),
    video_len = video_len,
    sample_frames=sample_frames,
    sample_offset=sample_offset,
    sample_mode="val",
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t


### model
model = dict(
    type=SAVi,
    encode_backbone=dict(
        type=CNN,
        in_dim=3,
        dims=[64, 64, 64, 64],
        kernels=[5, 5, 5, 5],
        strides=[1, 1, 1, 1],
        ctypes=[0, 0, 0, 0],
        gn=0,
        act="ReLU",
    ),
    encode_posit_embed=dict(
        type=CartesianPositionalEmbedding2d,
        resolut=resolut0,
        embed_dim=64,
    ),
    encode_project=dict(
        type=MLP, in_dim=64, dims=[vfm_dim, vfm_dim], act='relu', ln="pre",
    ),
    initializ=dict(type=SlotformerLatentsInit, num=slot_num, dim=slot_dim),
    projection=dict(
        type=GaussianProjection,
        slot_dim=slot_dim,
        latents_dist_layer=dict(type=Linear, in_features=slot_dim, out_features=slot_dim * 2),
    ),
    aggregat=dict(
        type=SlotAttention_rep,
        num_iter=2,
        embed_dim=slot_dim,
        ffn_dim=slot_dim * 2,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp=None,
    ),
    transit=dict(
        type=ResidualMLPTransit,
        ln=dict(type=LayerNorm, normalized_shape=slot_dim),
        mlp=dict(
            type=MLP, in_dim=slot_dim, dims=[slot_dim * 2, slot_dim], act='relu', ln=None
        ),
    ),
    decode=dict(
        type=BroadcastCNNDecoder,
        broadcast_resolut=broadcast_resolut,
        posit_embed=dict(
            type=CartesianPositionalEmbedding2d,
            resolut=broadcast_resolut,
            embed_dim=slot_dim,
        ),
        backbone=dict(
            type=CNN,
            in_dim=slot_dim,
            dims=[64, 64, 64, 64, 4],
            kernels=[5, 5, 5, 5, 1],
            strides=[2, 2, 2, 1, 1],
            ctypes=[1, 1, 1, 1, 0],
            gn=0,
            act="ReLU",
        ),
    ),
    clip_len = clip_len
)

model_imap = dict(video="batch.video")
model_omap = ["feature", "latents_dist", "slotz", "attent", "attent2", "recon"]

ckpt_map = []
freez = []


### learn
param_groups = None  # 占位
optimiz = dict(type=Adam, params=param_groups, lr=lr)  # 主逻辑在 train.py , param_groups = None 则 param_groups = model.parameters(); param_groups 有内容，则调用 ModelWrap.group_params()
gscale = dict(type=GradScaler)  # 梯度缩放
gclip = dict(type=ClipGradNorm, max_norm=0.05)  # 梯度裁剪


loss_fn = dict(
    kld_loss=dict(  # 确保来自高斯分布采样得到的 query 的方差保持在 0.01, 始终添加一定的随机噪音，保证两个 query 不会过于相似，从而避免这两个 slot 同时捕捉到新加入画面的同一个物体
        metric=dict(type=GaussianVarianceKLDLoss, dim=slot_dim, log_variance=math.log(0.01)),
        map=dict(input="output.latents_dist"),
        weight=1e-4,
    ),
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="batch.video"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
)

_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment2", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
_acc_dict_bgeh = dict(
    # metric=...,
    map=dict(input="output.segment2_bgeh", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> (b t) (h w) s"),  # "(b t) (h w) s"  for slotformer
    ),
)

acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    mbo_bgeh=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_bgeh),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    fari=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
    ari_bgeh=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_bgeh),
    fari_bgeh=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_bgeh),
    mbo_bgeh=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_bgeh),
    miou_bgeh=dict(metric=dict(type=mIoU, skip=[0]), **_acc_dict_bgeh),  # skip=[0] for align slotformer
)

before_step = [
    dict(
        type=Lambda, ikeys=[["batch.video", "batch.segment"]], func=move_to_cuda
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 40,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e2,
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attent2"]],  # (b,s,h,w) -> (b,h,w,s)
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment2"]],
    ),
    dict(
        type=Lambda,
        ikeys=[["output.attent2"]],  # (b,s,h,w) -> (b,h,w,s)
        func=lambda _: ptnf.one_hot(
            argmax_attent_bg_enhanced(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment2_bgeh"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
    dict(type=SaveCkpt, save_dir=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
]
callback_extract = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    dict(type=ExtractSlots, save_dir="./datasets/clevrer/SLOT-savi-clevrer_video"),
    callback_t[1],
]