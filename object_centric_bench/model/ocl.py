import torch as pt
import torch.nn as nn

from .basic import MLP


'''class SlotAttention(nn.Module):
    """TODO XXX modularization/cgv: correct the wrong implementation!"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]
        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)  # always bad for attention
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot.
        """
        b, n, c = query.shape
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query
        for _ in range(self_num_iter):
            if _ + 1 == self_num_iter:
                if self.trunc_bp == "bi-level":  # BO-QSA
                    q = q.detach() + query - query.detach()
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            z = self.norm2(y)
            q = y + self.ffn(z)  # droppath on ffn seems harmful
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key  #  todo diff2 eps=1e-6
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, eps=1e-5, h=4):
        q = rearrange(q, "b q (h d) -> (b h) q d", h=h)
        k = rearrange(k, "b k (h d) -> (b h) k d", h=h)
        v = rearrange(v, "b k (h d) -> (b h) k d", h=h)
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        o = rearrange(o, "(b h) q d -> b q (h d)", h=h)
        return o, a0'''


class SlotAttention_rep(nn.Module):
    """TODO XXX modularization/cgv: correct the wrong implementation!"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]
        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)  # always bad for attention
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim],act='relu', ln='pre', dropout=dropout)

    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot.
        """
        b, n, c = query.shape
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query
        for _ in range(self_num_iter):
            if _ + 1 == self_num_iter:
                if self.trunc_bp == "bi-level":  # BO-QSA
                    q = q.detach() + query - query.detach()
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            q = y + self.ffn(y)  # droppath on ffn seems harmful
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-6):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a0 =  a0 + eps
        a = a0 / (a0.sum(2, keepdim=True))  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0


class CartesianPositionalEmbedding2d(nn.Module):
    """"""

    def __init__(self, resolut: list, embed_dim: int):
        super().__init__()
        assert len(resolut) == 2
        self._pe = nn.Parameter(
            __class__.meshgrid(resolut)[None, ...], requires_grad=False
        )
        self.project = nn.Linear(4, embed_dim)

    @staticmethod
    def meshgrid(resolut, low=0, high=1):
        assert len(resolut) == 2
        yx = [pt.linspace(low, high, _) for _ in resolut]
        # yx = [(_[:-1] + _[1:]) / 2 for _ in yx]
        grid_y, grid_x = pt.meshgrid(*yx, indexing='ij')
        return pt.stack([grid_y, grid_x, 1 - grid_y, 1 - grid_x], 2)

    def forward(self, input):
        """
        input: in shape (b,h,w,c)
        output: in shape (b,h,w,c)
        """
        max_h, max_w = input.shape[1:3]
        output = input + self.project(self._pe[:, :max_h, :max_w, :])
        return output

    @property
    def pe(self):
        return self.project(self._pe)  # .flatten(1, -2)


class LearntPositionalEmbedding(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, resolut: list, embed_dim: int, in_dim: int = 0):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        if in_dim:
            self._pe = nn.Parameter(pt.zeros(1, *resolut, in_dim), requires_grad=True)
            self._project = nn.Linear(in_dim, embed_dim)
        else:
            self._pe = nn.Parameter(
                pt.zeros(1, *resolut, embed_dim), requires_grad=True
            )
        nn.init.trunc_normal_(self._pe)

    @property
    def pe(self):
        if hasattr(self, "_project"):
            return self._project(self._pe)
        return self._pe

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        # TODO XXX support variant length
        # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
        # pe = self.pe[:, :max_resolut, :]
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}"


# 相比与 NormalShared , NormalSeparat 引入了更强的随机性, 能更快收敛, 但稳定性并不一定优于 NormalShared
class NormalSeparat(nn.Module):
    """Separate gaussians as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, num, dim))
        self.logstd = nn.Parameter((pt.ones(1, num, dim) * dim**-0.5).log())  # (1, num, dim) 初始时所有值全部相同，经过学习后值会不同
        nn.init.xavier_uniform_(self.mean[0, :, :])  # 从均匀分布中采样, 不是 mean 中的所有值符合均匀分布，也不是 slot 之间或维度之间符合均匀分布, 而是多次运行 __init__ 后得到对应位置的值符合均匀分布, 不同位置之间是独立的

    def forward(self, b):
        smpl = self.mean.expand(b, -1, -1)
        if self.training:
            randn = pt.randn_like(smpl)   # 多次运行后, randn 对应位置的值属于标准高斯分布, 不同位置之间相互独立
            smpl = smpl + randn * self.logstd.exp()  # 多次运行后, smpl 对应位置的值属于高斯分布, 不同位置之间相互独立 (一组独立的高斯变量组成的张量), 随机的高斯噪音
        return smpl

    def extra_repr(self):
        return f"1, {self.num}, {self.dim}"


# NormalShared 与原版的 SA 基本一致, 只是这里的 mean 的先验为 Xavier Uniform 分布, SA 中的 mean 的先验为 N(0,1)
# 实际上在初始化阶段只是为了引入随机性，这些差异会在模型训练后, 参数更新时, 逐步减弱
class NormalShared(nn.Module):
    """Shared gaussian as queries."""

    # TODO new trick: Conditional Random Initialization

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, 1, dim))  # 与 NormalSeparat 的主要不同，所有 slot 共享一个均值和方差, 区别只来自于随机的高斯噪音
        self.logstd = nn.Parameter(pt.empty(1, 1, dim))  # 此时不仅仅是多次运行时 slot 对应的位置的值属于高斯分布，在同一次运行下每个 slot 之间对应位置值也符合高斯分布
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(self, b, n=None):
        self_num = self.num
        if n is not None:
            self_num = n
        smpl = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(smpl)
        smpl = smpl + randn * self.logstd.exp()
        return smpl


class SlotformerLatentsInit(nn.Module):
    """Separate gaussians as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.latents = nn.Parameter(pt.randn(1, num, dim))

    def forward(self, b):
        latents = self.latents.repeat(b, 1, 1)
        return latents



class LearntPositionalEmbedding_slotformer(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, resolut: list, embed_dim: int, in_dim: int = 0):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        if in_dim:
            self._pe = nn.Parameter(pt.zeros(1, *resolut, in_dim), requires_grad=True)
            self._project = nn.Linear(in_dim, embed_dim)
        else:
            self._pe = nn.Parameter(
                pt.zeros(1, *resolut, embed_dim), requires_grad=True
            )

    @property
    def pe(self):
        if hasattr(self, "_project"):
            return self._project(self._pe)
        return self._pe

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        # TODO XXX support variant length
        # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
        # pe = self.pe[:, :max_resolut, :]
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}"
