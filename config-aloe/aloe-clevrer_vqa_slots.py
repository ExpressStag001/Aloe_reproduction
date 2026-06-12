from object_centric_bench.datum import (
    Resize,
    Normalize,
    Lambda,
    CLEVRER_VQA_Slots,
    ClPadToMax1,
    ClevrerCollate,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    ClevrerVQAAccuracy,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveCkpt,
)
from object_centric_bench.model import (
    SAVi,
    Aloe,
    CNN,
    Embedding,
    MLP,
    SlotformerLatentsInit,
    GaussianProjection,
    SlotAttention_rep,
    LearntPositionalEmbedding_slotformer,
    TransformerEncoder,
    TransformerEncoderLayer,
    Linear,
    LayerNorm,
    CartesianPositionalEmbedding2d,
    BroadcastCNNDecoder,
    ResidualMLPTransit,
    CLEVRER_VQA_TransformerEncoder,
)

from object_centric_bench.util import Compose, ComposeNoStar, move_to_cuda

### global
slotz_config = "savi-clevrer_video"  # which SAVi config's slots to use
slot_num = 7
resolut0 = [64, 64]
broadcast_resolut = [8, 8]  # slot 空间广播输出的尺寸
slot_dim = 128
vfm_dim = 128

total_step = 240000 * 1  # scale with batch_size
val_interval = total_step // 40
batch_size_t = 256 // 1
batch_size_v = batch_size_t
num_work = 8  # todo
lr = 1e-3 / 1  # scale with batch_size

video_len = 128
sample_frames = 25  # 稀疏采样视频后的帧数
max_question_len = 20  # for padding
max_choice_len = 12  # for padding
vocab_q=82  # vocab 中 question 的种类数
vocab_a=22  # vocab 中 answer 的种类数
text_token_dim = 14  # 文本 token 嵌入维度
text_token_dim_plus = text_token_dim + 2 + 2  # 添加 text/video dim 以及 cls_q/mc_q/mc_c dim
vqa_tfe_nhead = 8
vqa_tfe_dim = text_token_dim_plus * vqa_tfe_nhead
vqa_tfe_token_len = slot_num * sample_frames + max_question_len + max_choice_len + 1  # L_video + L_q + L_c + L_CLS


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
    type=CLEVRER_VQA_Slots,
    base_dir=...,
    data_file="clevrer/train.lmdb",
    vocab_file="clevrer/vocab.json",
    slotz_file=f"clevrer/SLOT-{slotz_config}/slotz_train.lmdb",
    extra_keys=["question","segment"],
    pre_computed_keys=["slotz"],
    transform=dict(type=Compose, transforms=transform_t),
    video_len = video_len,
    max_question_len=max_question_len,
    max_choice_len=max_choice_len,
    sample_frames=sample_frames,
    sample_mode="train",
)
dataset_v = dict(
    type=CLEVRER_VQA_Slots,
    base_dir=...,
    data_file="clevrer/val.lmdb",
    vocab_file="clevrer/vocab.json",
    slotz_file=f"clevrer/SLOT-{slotz_config}/slotz_val.lmdb",
    extra_keys=["question","segment"],
    pre_computed_keys=["slotz"],
    transform=dict(type=Compose, transforms=transform_v),
    video_len = video_len,
    max_question_len=max_question_len,
    max_choice_len=max_choice_len,
    sample_frames=sample_frames,
    sample_mode="val",
)
dataset_test = dict(
    type=CLEVRER_VQA_Slots,
    base_dir=...,
    data_file="clevrer/test.lmdb",
    vocab_file="clevrer/vocab.json",
    slotz_file=f"clevrer/SLOT-{slotz_config}/slotz_test.lmdb",
    extra_keys=["question","segment"],
    pre_computed_keys=["slotz"],
    transform=dict(type=Compose, transforms=transform_v),
    video_len = video_len,
    max_question_len=max_question_len,
    max_choice_len=max_choice_len,
    sample_frames=sample_frames,
    sample_mode="val",
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=ClevrerCollate),
    ],
)
collate_fn_v = collate_fn_t


### model
model = dict(
    type=Aloe,
    slot_encoder=dict(
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
    ),
    vqa_head=dict(
        type=CLEVRER_VQA_TransformerEncoder,
        q_embedding=dict(type=Embedding, num_embeddings=vocab_q, embedding_dim=text_token_dim),
        project_q=dict(type=Linear, in_features=text_token_dim_plus, out_features=vqa_tfe_dim),
        project_v=dict(type=Linear, in_features=slot_dim + 2, out_features=vqa_tfe_dim),
        posit_embed=dict(
            type=LearntPositionalEmbedding_slotformer,
            resolut=[vqa_tfe_token_len],
            embed_dim=vqa_tfe_dim,
        ),
        backbone=dict(
            type=TransformerEncoder,
            encoder_layer=dict(
                type=TransformerEncoderLayer,
                d_model=vqa_tfe_dim,
                nhead=vqa_tfe_nhead,
                dim_feedforward=512,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=12,
            norm=None,
        ),
        mlp_a_cls=dict(
            type=MLP, in_dim=vqa_tfe_dim, dims=[vfm_dim, vocab_a], act='relu', ln=None
        ),
        mlp_a_mc=dict(
            type=MLP, in_dim=vqa_tfe_dim, dims=[vfm_dim, 1], act='relu', ln=None
        ),
        d_model=vqa_tfe_dim,
        mc_question_len=max_question_len,
        slot_dim=slot_dim,
        mask_obj_loss=False,  # don't use `mask_obj_loss` when using SAVi slots
    ),
    use_pre_computed=True,
)

model_imap = dict(video="batch.video", question="batch.question", pre_computed="batch.pre_computed")
model_omap = ["feature", "latents_dist", "slotz", "attent", "attent2", "recon", "vqa_a_dict"]

ckpt_map = []
freez = []


### learn
param_groups = None  # 占位
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)  # 梯度缩放
gclip = None  # 梯度裁剪

loss_fn = dict(
    cls_answer_loss=dict(
        metric=dict(type=CrossEntropyLoss),
        map=dict(input="output.vqa_a_dict.cls_answer_logits", target="batch.question.cls_label"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    mc_answer_loss=dict(
        metric=dict(type=BinaryCrossEntropyLoss),
        map=dict(input="output.vqa_a_dict.mc_answer_logits", target="batch.question.mc_label"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach().float()),
    ),
)

_acc_vqa_cls_dict_ = dict(
    map=dict(input="output.vqa_a_dict.cls_answer_logits", target="batch.question.cls_label"),
)
_acc_vqa_mc_dict_ = dict(
    map=dict(input="output.vqa_a_dict.mc_answer_logits", target="batch.question.mc_label", mc_flag="batch.question.mc_flag", mc_subtype="batch.question.mc_subtype"),
)
acc_fn_t = dict(
    cls_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=0, subtype=None),**_acc_vqa_cls_dict_),
    mc_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=1, subtype=None),**_acc_vqa_mc_dict_),
)
acc_fn_v = dict(
    cls_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=0, subtype=None), **_acc_vqa_cls_dict_),
    mc_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=1, subtype=None), **_acc_vqa_mc_dict_),
    mc_exp_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=1, subtype=1),**_acc_vqa_mc_dict_),
    mc_pred_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=1, subtype=2),**_acc_vqa_mc_dict_),
    mc_cfc_acc=dict(metric=dict(type=ClevrerVQAAccuracy, q_type=1, subtype=3),**_acc_vqa_mc_dict_),
)
before_step = [
    dict(
        type=Lambda, ikeys=[["batch.video", "batch.segment", "batch.question", "batch.pre_computed"]], func=move_to_cuda
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 10,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e2,
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, ),  # after_forward=after_forward
    dict(type=AverageLog, log_file=...),
    dict(type=SaveCkpt, save_dir=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], ),  # after_forward=after_forward
    callback_t[1],
]