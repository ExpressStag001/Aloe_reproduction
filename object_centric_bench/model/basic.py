import re

from einops import rearrange
import timm
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
from safetensors.torch import load_file
from timm.layers.pos_embed import resample_abs_pos_embed

from ..util import DictTool


####


class ModelWrap(nn.Module):  # TODO XXX TensorDictModule

    def __init__(self, m: nn.Module, imap, omap):
        """
        - imap: dict or list.
            If keys in batch mismatches with keys in model.forward, use dict, ie, {key_in_batch: key_in_forward};
            If not, use list.
        - omap: list
        """
        super().__init__()
        assert isinstance(imap, (dict, list, tuple))
        assert isinstance(omap, (list, tuple))
        self.m = m
        self.imap = imap if isinstance(imap, dict) else {_: _ for _ in imap}
        self.omap = omap

    # def forward(self, input: dict) -> dict:
    def forward(self, **pack: dict) -> dict:
        # input2 = {k: input[v] for k, v in self.imap.items()}
        input2 = {k: DictTool.getattr(pack, v) for k, v in self.imap.items()}
        output = self.m(**input2)
        if not isinstance(output, (list, tuple)):
            output = [output]
        assert len(self.omap) == len(output)
        output2 = dict(zip(self.omap, output))
        return output2

    def load(self, ckpt_file: str, ckpt_map: list, verbose=True):
        state_dict = pt.load(ckpt_file, map_location="cpu", weights_only=True)
        if ckpt_map is None:
            if verbose:
                print("fully")
            self.load_state_dict(state_dict)  # TODO XXX , False
        elif isinstance(ckpt_map, (list, tuple)):
            for dst, src in ckpt_map:
                dkeys = [_ for _ in self.state_dict() if _.startswith(dst)]
                skeys = [_ for _ in state_dict if _.startswith(src)]
                assert len(dkeys) == len(skeys)  # > 0
                if len(dkeys) == 0:
                    print(
                        f"[{__class__.__name__}.load WARNING] ``{dst}, {src}`` has no matched keys !!!"
                    )
                for dk, sk in zip(dkeys, skeys):
                    if verbose:
                        print(dk, sk)
                    self.state_dict()[dk].data[...] = state_dict[sk]
        else:
            raise "ValueError"
        if verbose:
            print(f"checkpoint ``{ckpt_file}`` loaded")

    def save(self, save_file, weights_only=True, key=r".*"):
        if weights_only:
            save_obj = self.state_dict()
            save_obj = {k: v for k, v in save_obj.items() if re.match(key, k)}
        else:
            save_obj = self
        pt.save(save_obj, save_file)

    def freez(self, freez: list, verbose=True):
        for n, p in self.named_parameters():
            for f in freez:
                if bool(re.match(f, n)):
                    p.requires_grad = False
        if verbose:
            [print(k, v.requires_grad) for k, v in self.named_parameters()]

    def group_params(self, coarse=r"^.*", fine=dict()):
        """Group model parameters by coarse and fine filters.

        - coarse: coarse filter; regex string
        - fine: fine filter for grouping and adding extras; {regex1: dict(lr_mult=0.5, wd_mult=0),..}
        """
        # coarse filtering
        named_params = dict(self.named_parameters())
        named_params = {
            k: v for k, v in named_params.items() if bool(re.match(coarse, k))
        }
        if not fine:
            params = []
            for k, v in named_params.items():
                if v.requires_grad:
                    print(f"{k} - to train, require grad")
                    params.append(v)
                else:
                    print(f"{k} - skipped, not require grad")
            return params

        # fine filtering
        param_groups = {k: dict(params=[]) for k in fine}  # TODO lr
        names = list(named_params.keys())
        for n, p in named_params.items():
            for g, (k, v) in enumerate(fine.items()):
                assert isinstance(v, dict)
                if bool(re.match(k, n)):
                    cursor = names.pop(0)
                    assert cursor == n  # ensure no missing or overlap
                    if p.requires_grad:
                        print(f"{n} - #{g}, {v}")
                        param_groups[k]["params"].append(p)
                        param_groups[k].update(v)
                    else:
                        print(f"{n} - #{g}, skipped, not require grad")

        param_groups = {k: v for k, v in param_groups.items() if len(v["params"])}
        return list(param_groups.values())


class Sequential(nn.Sequential):
    """"""

    def __init__(self, modules: list):
        super().__init__(*modules)

    def forward(self, input):
        for module in self:
            if isinstance(input, (list, tuple)):  # TODO control in init
                input = module(*input)
            else:
                input = module(input)
        return input


ModuleList = nn.ModuleList


####


Embedding = nn.Embedding


Conv2d = nn.Conv2d


PixelShuffle = nn.PixelShuffle


ConvTranspose2d = nn.ConvTranspose2d


AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d


Identity = nn.Identity


ReLU = nn.ReLU


GELU = nn.GELU


SiLU = nn.SiLU


Mish = nn.Mish


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, interp="bilinear"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.interp = interp

    def forward(self, input):
        return ptnf.interpolate(input, self.size, self.scale_factor, self.interp)
    # ptnf.interpolate 的参数介绍
    # input 必须是形状为 [B, C, H, W] 的张量
    # size 直接指定输出的尺寸 (H_out, W_out)
    # scale_factor 指定放大 / 缩小的倍数，不能与 size 同时使用
    # mode 可选['bilinear', 'bicubic', 'nearest']. 'bilinear'和'bicubic' 适用于连续型数据，如图像，'bicubic'精度高计算慢; 'nearest' 适用于离散型数据，如 mask

class Conv2dPixelShuffle(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        upscale=2,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels * upscale**2,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        shuff = nn.PixelShuffle(upscale)
        super().__init__(conv, shuff)

Dropout = nn.Dropout


Linear = nn.Linear


GroupNorm = nn.GroupNorm


LayerNorm = nn.LayerNorm


####


MultiheadAttention = nn.MultiheadAttention


TransformerEncoderLayer = nn.TransformerEncoderLayer


TransformerDecoderLayer = nn.TransformerDecoderLayer


TransformerEncoder = nn.TransformerEncoder


TransformerDecoder = nn.TransformerDecoder


###
# class CNN(nn.Sequential):
#     """hyperparam setting of ConvTranspose2d:
#     https://blog.csdn.net/pl3329750233/article/details/130283512.
#     """
#
#     conv_types = {
#         0: nn.Conv2d,  # nn.Conv2d 普通卷积 (用于encoder)
#         1: lambda *a, **k: nn.ConvTranspose2d(*a, **k, output_padding=stride-1),  # nn.ConvTranspose2d 反卷积 (用于decoder)
#         2: lambda *a, **k: Conv2dPixelShuffle(*a, **k, upscale=2),  # nn.Conv2d + PixelShuffle 上采样 (用于decoder)
#     }
#
#     def __init__(self, in_dim, dims, kernels, strides, ctypes=0, gn=0, act="SiLU"):
#         """
#         - ctypes: 0 for normal conv2d, 1 for convtransposed, 2 for convpixelshuffle
#         - gn: 0 for no groupnorm, >0 for groupnorm(num_groups=g)
#         """
#         if isinstance(ctypes, int):
#             ctypes = [ctypes] * len(dims)
#         assert len(dims) == len(kernels) == len(strides) == len(ctypes)
#         num = len(dims)
#
#         layers = []
#         ci = in_dim
#
#         for i, (t, c, k, s) in enumerate(zip(ctypes, dims, kernels, strides)):
#             p = k // 2 if k % 2 != 0 else 0  # XXX for k=s=4, requires isize%k==0
#             # 填充 p = k // 2 能在步长为 1 的前提下，不改变分辨率。但当这里 p = 0 时，分辨率会改变
#             if i + 1 < num:
#                 block = [
#                     __class__.conv_types[t](ci, c, k, stride=s, padding=p),
#                     nn.GroupNorm(gn, c) if gn else None,
#                     nn.__dict__[act](inplace=True),  # SiLU>Mish>ReLU>Hardswish
#                 ]
#             else:  # 最后一层
#                 block = [
#                     __class__.conv_types[t](ci, c, k, stride=s, padding=p),
#                 ]
#
#             layers.extend([_ for _ in block if _])
#             ci = c
#
#         super().__init__(*layers)

class CNN(nn.Sequential):

    def __init__(self, in_dim, dims, kernels, strides, ctypes=0, gn=0, act="SiLU"):
        """
        conv_types = {
            0: "conv",
            1: "deconv",
            2: "pixelshuffle",
        }
        - gn: 0 for no groupnorm, >0 for groupnorm(num_groups=g)
        """
        if isinstance(ctypes, int):
            ctypes = [ctypes] * len(dims)
        assert len(dims) == len(kernels) == len(strides) == len(ctypes)
        num = len(dims)

        layers = []
        ci = in_dim

        for i, (t, c, k, s) in enumerate(zip(ctypes, dims, kernels, strides)):
            p = k // 2 if k % 2 != 0 else 0

            if t == 0:  # Conv2d
                conv = nn.Conv2d(ci, c, k, stride=s, padding=p)
            elif t == 1:  # ConvTranspose2d
                conv = nn.ConvTranspose2d(ci, c, k, stride=s, padding=p, output_padding=s-1)
            elif t == 2:  # PixelShuffle
                conv = Conv2dPixelShuffle(ci, c, k, stride=s, padding=p, upscale=2)
            else:
                raise ValueError(f"Unknown conv type {t}")

            if i + 1 < num:
                block = [
                    conv,
                    nn.GroupNorm(gn, c) if gn else None,
                    nn.__dict__[act](inplace=True),  # SiLU>Mish>ReLU>Hardswish
                ]
            else:  # 最后一层
                block = [conv]

            layers.extend([_ for _ in block if _])
            ci = c

        super().__init__(*layers)


class MLP(nn.Sequential):
    """"""

    def __init__(self, in_dim, dims, act: str = "gelu", ln: str = None, dropout=0):
        """
        - ln: None for no layernorm, 'pre' for pre-norm, 'post' for post-norm
        """
        assert ln in [None, "pre", "post"]
        assert act in ["gelu", "relu", "silu"]

        if act == "gelu":
            act_layer = nn.GELU
        elif act == "relu":
            act_layer = nn.ReLU
        elif act == "silu":
            act_layer = nn.SiLU

        num = len(dims)
        layers = []
        ci = in_dim

        if ln == "pre":
            layers.append(nn.LayerNorm(ci))

        for i, c in enumerate(dims):
            if i + 1 < num:
                layers.append(nn.Linear(ci, c))
                layers.append(act_layer())
                if dropout:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Linear(ci, c))
            ci = c

        if ln == "post":
            layers.append(nn.LayerNorm(ci))

        super().__init__(*layers)



class DINO(nn.Module):
    def __init__(
            self,
            dino_name='vit_small_patch14_dinov2.lvd142m',
            layers_number=["blocks.11"], # 6，9，11
            dino_parameter_path='default',
            in_size=518,
            rearrange=True,
            norm_out=True,
    ):
        """
            Args:
                dino_name: 所用的 dino 模型名，见 dino_name_pool
                layers_number: dino特征层的序号，格式为字符串列表 ["blocks.n"] ，比如只提取第一层，第二层的特征 layers_number=["blocks.0", "blocks.1"]
                dino_parameter_path: 本地参数的路径
                rearrange: 是否去掉 CLS token 并使得 "b (h w) c -> b c h w"
                norm_out: 是否对输出的特征进行归一化
            """

        super().__init__()
        self.in_size = in_size
        self.rearrange = rearrange
        self.norm_out = norm_out

        if dino_parameter_path == 'default':
            dino_parameter_path = f"../dino_parameters/{dino_name}.safetensors"

        self.model = timm.create_model(dino_name, pretrained=False, img_size=self.in_size)

        cls_tokens = getattr(self.model, 'cls_token', None)
        reg_tokens = getattr(self.model, 'reg_token', None)  # 尝试从模型中取出 reg_token 这个属性，如果模型中没有这个属性，就返回 None
        num_cls = 1 if cls_tokens is not None else 0
        num_reg = reg_tokens.shape[1] if reg_tokens is not None else 0  # reg_tokens 形状为 (B, num_reg, feature_dim)
        self.num_prefix_tokens = num_cls + num_reg
        self.patch_size = self.model.patch_embed.patch_size[0]

        dino_state_dict = load_file(dino_parameter_path)  # safetensor 文件用 load_file加载

        # 将 DINO 预训练权重中的绝对位置编码 pos_embed 插值到“当前模型 patch 网格数量”所需要的长度上
        # 仅 dinov2 需要执行下方 if 中的代码, dinov3 不需要执行(权重中无 pos_embed 字段)
        if 'pos_embed' in dino_state_dict:
            pos_embed = dino_state_dict['pos_embed'].clone()  # [1, 1369, C]
            print(pos_embed.shape[1])
            old_hw = int(pos_embed.shape[1]**0.5)
            old_hw = (old_hw, old_hw)
            new_hw = (self.in_size//self.patch_size, self.in_size//self.patch_size)  # 新 patch grid 尺寸
            dino_state_dict['pos_embed'] = resample_abs_pos_embed(
                pos_embed,
                new_size=new_hw,
                old_size=old_hw,
                num_prefix_tokens=0,  # 经测试 dinov2, 普通版本此处填 1, reg4 版此处填 0.
            )
        self.model.load_state_dict(dino_state_dict, strict=False)

        self.norm = self.model.norm if norm_out else None

        print(f"[DINO] Transformer depth = {len(self.model.blocks)}")
        if isinstance(layers_number, (list, tuple)):
            self.layers_list = list(layers_number)
        else:
            self.layers_list = [layers_number]

        # module_names = [name for name, m in model.named_modules()]
        # model.named_modules() 是一个nn.Module定义的可迭代对象，对其遍历可获取模型的各个模块名称 name 和对应的模块实例 m
        # module_names 记录了完整的 vit 各部件的名称，从patch_embed(ptach嵌入)、pos_drop(位置编码)--> blocks.n(第n层整体的输出)、blocks.n.attn(第n层自注意力后的输出)、blocks.n.mlp(第n层经过mlp后的输出)。对于本次任务只需关注blocks.n

        self.layers_feature_dict = {}  # 预定义一个字典，用于接收 hook 中返回的layer特征

        def make_hook(layer_name):
            # 定义钩子函数，它的三个参数由register_forward_hook自动传递
            def hook(module, inputs, outputs):
                self.layers_feature_dict[layer_name] = outputs.detach().clone()
            return hook

        for layer_name in self.layers_list:
            layer = dict(self.model.named_modules())[layer_name]  # 将可迭代对象转为字典后，通过模块名称layers_number，获取模块实例layer(实际上也是nn.Module的一个实例)
            layer.register_forward_hook(make_hook(layer_name))  # register_forward_hook会自动调用 hook 函数


    def forward(self,image):
        """
        input:
            image.shape=(b,c,H,W), float. 图像尺寸可以通过 self.transforms 自动对齐

        output:
            layers_feature_dict_p, dict { key(layer_name) : value(layer_feature)}, layer_feature.shape = (b c h w)
        """

        self.layers_feature_dict.clear()
        assert image.shape[2] == self.in_size and image.shape[3] == self.in_size
        assert image.shape[2] % self.patch_size == 0 and image.shape[3] % self.patch_size == 0  # 确保 image 的长宽能整除 patch_size

        with pt.no_grad():
            _ = self.model(image)  # 调用 model 中的 forward，是为了触发 register_forward_hook

        layers_feature_dict_p = {}
        for layer_name in self.layers_list:
            layer_feature = self.layers_feature_dict[layer_name]

            if self.norm_out:
                layer_feature = self.norm(layer_feature)

            if self.rearrange:
                layer_feature = layer_feature[:, self.num_prefix_tokens:, :]  # remove class token
                assert layer_feature.shape[1] == (image.shape[2] // self.patch_size) * (image.shape[3] // self.patch_size)
                layer_feature = rearrange(layer_feature, "b (h w) c -> b c h w", h=image.shape[2]//self.patch_size)

            layers_feature_dict_p[layer_name] = layer_feature
        return layers_feature_dict_p



# https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8
dinov2_name_pool = [
    'vit_small_patch14_dinov2.lvd142m',
    'vit_large_patch14_dinov2.lvd142m',

    # reg4 版本, reg4 表示有 4 个 reg_token
    'vit_small_patch14_reg4_dinov2.lvd142m',
    'vit_base_patch14_reg4_dinov2.lvd142m',
    'vit_large_patch14_reg4_dinov2.lvd142m',
    'vit_giant_patch14_reg4_dinov2.lvd142m',
]


dinov3_name_pool = [
    'vit_small_patch16_dinov3.lvd1689m',
    'vit_base_patch16_dinov3.lvd1689m',
    'vit_large_patch16_dinov3.lvd1689m',
    'vit_7b_patch16_dinov3.lvd1689m',

    # plus 版本, 表示由 7b 版本蒸馏得到的 small、huge 版本
    'vit_small_plus_patch16_dinov3.lvd1689m',
    'vit_huge_plus_patch16_dinov3.lvd1689m',

    # qkvb 版本, 表示保留 偏置(bias) 参数, 但实际上偏置均为0, 仅为了对齐 vit. 普通版本将 bias 参数删除，节约空间
    'vit_small_patch16_dinov3_qkvb.lvd1689m',

    # sat493m 后缀, 表示卫星图像数据集训练
    'vit_large_patch16_dinov3.sat493m',

    # convnext 版本, 框架使用了基于 CNN 的 convnext, 而非 vit
    'convnext_tiny.dinov3_lvd1689m',
    'convnext_small.dinov3_lvd1689m',
    'convnext_base.dinov3_lvd1689m',
    'convnext_large.dinov3_lvd1689m',
]

download_dino_name_pool = [
    'vit_small_patch14_dinov2.lvd142m',
    'vit_small_patch14_reg4_dinov2.lvd142m',
    'vit_small_patch16_dinov3.lvd1689m',
    'vit_base_patch16_dinov3.lvd1689m',
    'vit_small_plus_patch16_dinov3.lvd1689m',
    'convnext_small.dinov3_lvd1689m',
]
