from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class SAVi(nn.Module):
    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        projection,
        aggregat,
        transit,
        decode,
        clip_len=32,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.projection = projection
        self.aggregat = aggregat
        self.transit = transit
        self.decode = decode
        self.clip_len = clip_len
        self.reset_parameters(
            [self.encode_backbone, self.encode_posit_embed, self.encode_project, self.initializ, self.aggregat, self.transit, self.decode]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def _forward(self, video, condit=None):
        """
        - video: shape=(b,t,c,h,w)
        - condit: condition, shape=(b,n,c)
        """
        b, t, c, h, w = video.shape
        video = video.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(video)  # (b*t,c,h,w)  注意：dino 这里才需要 .detach()
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        if condit is None:
            latents = self.initializ(b)  # (b,n,c)
        else:
            latents = self.transit(condit)

        slotz = []
        attent = []
        latents_dist = []

        for i in range(t):
            query, latents_dist_i = self.projection(latents)
            slotz_i, attent_i = self.aggregat(encode[:, i, :, :], query)
            latents = self.transit(slotz_i)

            latents_dist.append(latents_dist_i)  # [(b,n,2c),..]
            slotz.append(slotz_i)  # [(b,n,c),..]
            attent.append(attent_i)  # [(b,n,h*w),..]

        latents_dist = pt.stack(latents_dist, 1)  # (b,t,n,2c)
        slotz = pt.stack(slotz, 1)  # (b,t,n,c)
        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        recon, attent2 = self.decode(slotz.flatten(0, 1))
        recon = rearrange(recon, "(b t) c h w -> b t c h w", b=b)
        attent2 = rearrange(attent2, "(b t) n h w -> b t n h w", b=b)

        return feature, latents_dist, slotz, attent, attent2, recon


    def forward(self, video, condit=None):
        T = video.shape[1]

        if self.training:
            assert T <= self.clip_len
        if T <= self.clip_len:
            return self._forward(video, condit)

        feature = []
        latents_dist = []
        slotz = []
        attent = []
        attent2 = []
        recon = []
        for clip_start in range(0, T, self.clip_len):
            clip_end = min(clip_start + self.clip_len, T)
            clip = video[:, clip_start:clip_end]
            with pt.inference_mode():
                feat, ld, sz, att, att2, rec = self._forward(clip, condit)

            feature.append(feat)
            latents_dist.append(ld)
            slotz.append(sz)
            attent.append(att)
            attent2.append(att2)
            recon.append(rec)

            condit = sz[:, -1] # (b, n, c)

        feature = pt.cat(feature, dim=1)
        latents_dist = pt.cat(latents_dist, dim=1)
        slotz = pt.cat(slotz, dim=1)
        attent = pt.cat(attent, dim=1)
        attent2 = pt.cat(attent2, dim=1)
        recon = pt.cat(recon, dim=1)

        return feature, latents_dist, slotz, attent, attent2, recon


class GaussianProjection(nn.Module):
    def __init__(self, slot_dim, latents_dist_layer):
        super().__init__()
        self.slot_dim = slot_dim
        self.latents_dist_layer = latents_dist_layer

    def forward(self, latents):
        latents_dist = self.latents_dist_layer(latents)

        assert latents_dist.shape[-1] == self.slot_dim * 2
        mu = latents_dist[..., :self.slot_dim]
        log_var = latents_dist[..., self.slot_dim:]
        query = mu + pt.randn_like(mu).detach() * pt.exp(log_var * 0.5)

        return query, latents_dist


class ResidualMLPTransit(nn.Module):
    """LN + residual MLP."""

    def __init__(self, ln, mlp):
        super().__init__()
        self.ln = ln
        self.mlp = mlp

    def forward(self, slots):
        x = self.ln(slots)
        return self.mlp(x) + x


class BroadcastCNNDecoder(nn.Module):
    """SAVi's decoder."""

    def __init__(self, broadcast_resolut, posit_embed, backbone):
        super().__init__()
        self.broadcast_resolut = broadcast_resolut
        self.posit_embed = posit_embed
        self.backbone = backbone

    def forward(self, slotz):
        """
        - slotz: slots, shape=(b,n,c)
        """
        h, w = self.broadcast_resolut
        b, n, c = slotz.shape

        mixture = repeat(slotz, "b n c -> (b n) h w c", h=h, w=w)
        mixture = self.posit_embed(mixture)  # (b*n,h,w,c)
        mixture = mixture.permute(0, 3, 1, 2)  # (b*n,c,h,w)
        mixture = self.backbone(mixture)  # (b*n,c=3+1,16*h,16*w)

        recon, alpha = mixture[:, :-1, :, :], mixture[:, -1:, :, :]
        recon = rearrange(recon, "(b n) c h w -> b n c h w", b=b)
        alpha = rearrange(alpha, "(b n) 1 h w -> b n 1 h w", b=b)
        # faster than pt.einsum()

        alpha = alpha.softmax(1)
        recon = (recon * alpha).sum(1)  # (b,c,h,w)
        attent2 = alpha[:, :, 0, :, :]  # (b,n,h,w)

        return recon, attent2