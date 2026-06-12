from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class Aloe(nn.Module):
    def __init__(
        self,
        vqa_head,
        slot_encoder=None,
        use_pre_computed=False,
    ):
        super().__init__()
        self.slot_encoder = slot_encoder
        self.vqa_head = vqa_head
        self.use_pre_computed = use_pre_computed

    def forward(
        self, video, question, condit=None, pre_computed=None,
    ):
        """
        Mode 1: End-to-End
            - video: shape=(b,t,c,h,w)
            - question: dict
            - condit: condition, shape=(b,t,n,c)

        Mode 2: Pre-computed
            - video: shape=(b,t,c,h,w)
            - question: dict
            - condit: condition, shape=(b,t,n,c)
            - pre_computed:{
                - slotz: (b,t,n,c)
                - attent2: (b,t,n,h,w)  optional
            }
        """

        if self.use_pre_computed:
            assert pre_computed is not None
            slotz = pre_computed["slotz"]

            feature = pre_computed.get("feature", None)
            latents_dist = pre_computed.get("latents_dist", None)
            attent = pre_computed.get("attent", None)
            attent2 = pre_computed.get("attent2", None)
            recon = pre_computed.get("recon", None)

        else:
            assert self.slot_encoder is not None
            feature, latents_dist, slotz, attent, attent2, recon = self.slot_encoder(video, condit)

        vqa_a_dict = self.vqa_head(slotz.detach(), question)  # slotz.detach() 决定了不同 loss 的作用域

        return feature, latents_dist, slotz, attent, attent2, recon, vqa_a_dict


class CLEVRER_VQA_TransformerEncoder(nn.Module):
    def __init__(
            self,
            q_embedding,
            project_q,
            project_v,
            posit_embed,
            backbone,
            mlp_a_cls,
            mlp_a_mc,

            d_model = 144,
            mc_question_len = 20,
            slot_dim = 128,
            mask_obj_loss=True,
    ):
        super().__init__()

        self.q_embedding = q_embedding
        self.project_q = project_q
        self.project_v = project_v
        self.posit_embed = posit_embed
        self.backbone = backbone
        self.mlp_a_cls = mlp_a_cls
        self.mlp_a_mc = mlp_a_mc

        self.d_model = d_model
        self.mc_question_len = mc_question_len
        self.slot_dim = slot_dim
        self.mask_obj_loss = mask_obj_loss

        self.CLS = nn.Parameter(pt.zeros(1, 1, self.d_model))

        self.text_token = nn.Parameter(
            pt.tensor([1, 0]).float(), requires_grad=False)
        self.vision_token = nn.Parameter(
            pt.tensor([0, 1]).float(), requires_grad=False)

        self.cls_token = nn.Parameter(
            pt.tensor([0, 1]).float(), requires_grad=False)
        self.mc_question_token = nn.Parameter(
            pt.tensor([1, 0]).float(), requires_grad=False)
        self.mc_choice_token = nn.Parameter(
            pt.tensor([0, 1]).float(), requires_grad=False)

        if self.mask_obj_loss:
            self.mask_token = nn.Parameter(pt.zeros((1, slot_dim)))  # learnable [MASK] token
            self.mask_obj_fc = nn.Linear(self.d_model, slot_dim)


    def _mask_v_embedding(self, v_embedding, mask_token):
        """
        random mask one position per frame.
         Args:
            v_embedding: [B, T, N, C]
            mask_token: [1, C]

        Returns:
            v_embedding: [B, T, N, C] ,after masking
            mask_idx: [B * T,]
            gt_v_emb: [B*T, C]
        """

        B, T, N, C = v_embedding.shape
        v_flat = rearrange(v_embedding, 'b t n c -> (b t) n c')
        mask_idx = pt.randint(0, N, (B * T,)).to(v_flat.device)  # (B * T,), 从整数集合 {0, 1, 2, ..., N-1} 中随机采样 B*T 个整数，形成一个形状为 (B * T,) 的张量
        batch_idx = pt.arange(B * T).type_as(mask_idx).to(v_flat.device)  # 输出tensor([0, 1, 2, 3 ... B * T])

        gt_v_emb = v_flat[batch_idx, mask_idx].detach()  # [B*T, C], 每一帧都会随机 mask 掉一个 token/slot, gt_v_emb 存储着被 mask 掉的真实 token, 以 idx 为索引从 v_embedding 中提取值
        v_flat[batch_idx, mask_idx] = mask_token.repeat(len(gt_v_emb), 1).type_as(gt_v_emb)  # 用 mask_token 取代被 mask 掉的 token，mask_token 为初值为 0 的可学习张量
        v_embedding = rearrange(v_flat, '(b t) n c -> b t n c', b=B)

        return v_embedding, mask_idx, gt_v_emb


    def _process_in_embeddings(self, v_embedding, q_embedding, q_pad_mask):
        """Prepare input for Transformer.

        Args:
            v_embedding: [B, T, N, C1]
            q_embedding: [B, L, C2+2]
            q_pad_mask: [B, L]

        Returns:
            in_embedding: [B, 1 + T*N + L, C]
            pad_mask: [B, 1 + T*N + L]
            mask_idx: [B * T]
            gt_v_emb: [B * T, C1], one emb per frame
        """

        b = q_embedding.shape[0]

        # according to Aloe, we mask one object per timestep
        if self.mask_obj_loss:
            v_embedding, mask_idx, gt_v_emb = self._mask_v_embedding(v_embedding, self.mask_token) # 将每一帧中随机一个 token/slot mask 掉
        else:
            mask_idx, gt_v_emb = None, None

        # unroll along temporal dim
        v_embedding = rearrange(v_embedding, 'b t n c -> b (t n) c')  # [B, T*N, C1]
        v_embedding = self.batch_cat_vec(v_embedding, self.vision_token, dim=-1)  # [B, T*N, C1+2]
        v_embedding = self.project_v(v_embedding)  # [B, T*N, C]

        q_embedding = self.batch_cat_vec(q_embedding, self.text_token, dim=-1)  # [B, L, C2+2+2]
        q_embedding = self.project_q(q_embedding)  # [B, L, C]

        CLS = self.CLS.repeat(b, 1, 1)  # [B, 1, d_model]
        in_embedding = pt.cat([CLS, v_embedding, q_embedding], dim=1)  # [B, 1 + T*N + L, C]

        # q_pad_mask 的形状为(B, L), 是仅属于 question 文本的 mask
        # pad_mask 的形状为(B, 1 + T*N + L), 即为整个 in_embedding 的 mask, “construct padding mask, CLS and vision tokens should be False”
        # pad_mask 将在 transformer 阶段应用于 in_embedding
        no_pad_mask = pt.zeros(b, in_embedding.shape[1] - q_pad_mask.shape[1]).type_as(q_pad_mask)  # [B, 1 + T*N]
        pad_mask = pt.cat([no_pad_mask, q_pad_mask], dim=-1)

        return in_embedding, pad_mask, mask_idx, gt_v_emb


    def _cls_forward(self, video_emb, question):
        """
        Args:
            video_emb: [b, t, n, c]

            question:{
                    - question_type: Tensor, (b,)

                    - cls_q_tokens: Tensor, (b1, L,)
                    - cls_q_pad_mask: Tensor, (b1, L,)
                    - cls_label: Tensor, (b1,)
                }

        Returns:
            answer_logits : [B1, num_classes]
            gt_v_emb: [B1 * T, C1]
            pred_v_emb: [B1 * T, C1]
        """
        # no cls question in this batch
        if (question['question_type'] == 0).sum() == 0:
            return None

        v_embedding = video_emb[question['question_type'] == 0]  # [b1, t, n, C1]
        B, T, N, _ = v_embedding.shape

        q_tokens, q_pad_mask = question['cls_q_tokens'], question['cls_q_pad_mask']
        q_embedding = self.q_embedding(q_tokens)  # [B, L, C2]，查表操作
        q_embedding = self.batch_cat_vec(q_embedding, self.cls_token, dim=-1)   # [B, L, C2 + 2] ，将 cls 标签拼接到 q_embedding 中

        in_embedding, pad_mask, mask_idx, gt_v_emb = self._process_in_embeddings(v_embedding, q_embedding, q_pad_mask)  # 将 v_embedding 和 q_embedding 进行维度对齐、 mask 操作、添加 CLS toekn 后 ，嵌入为一个整体

        in_embedding = self.posit_embed(in_embedding)
        transformer_out = self.backbone(in_embedding, src_key_padding_mask=pad_mask)  # [B, 1 + T*N + L, d_model]

        # multi-class classification
        cls_emb = transformer_out[:, 0, :]  # 提取 CLS token
        answer_logits = self.mlp_a_cls(cls_emb)  # [B, num_classes]

        # masked object prediction
        if self.mask_obj_loss:
            out_v_emb = transformer_out[:, 1:1 + T * N].reshape(B * T, N, -1)  # 该 out_v_emb 来自已被 mask 的 in_embedding，但这里它已经将 “被 mask 区域” 通过 self—attention 补齐了
            batch_idx = pt.arange(B * T, device=out_v_emb.device)
            mask_v_emb = out_v_emb[batch_idx, mask_idx]  # [B*T, C], 提取 self—attention 补齐后的 mask 区域
            pred_v_emb = self.mask_obj_fc(mask_v_emb)  # [B * T, C1]
        else:
            pred_v_emb = None

        return {
            'answer_logits': answer_logits,
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }


    def _mc_forward(self, video_emb, question):
        """
        Args:
            video_emb: [b, t, n, c]

            question:{
                - question_type: Tensor, (b,)

                - mc_subtype: Tensor, (b2, )
                - mc_q_tokens: Tensor, (b2n, L), concated along num_choices dim
                - mc_q_pad_mask: Tensor, (b2n, L)
                - mc_label: Tensor, (b2n,)
                - mc_flag: Tensor, (b2n,) e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
            }

        Returns:
            answer_logits : [B2n]
            gt_v_emb: [B2n * T, C1]
            pred_v_emb: [B2n * T, C1]
        """

        # no mc question in this batch
        if (question['question_type'] == 1).sum() == 0:
            return None

        # repeat v_embedding to pair up with each question
        v_embedding = video_emb[question['question_type'] == 1]  # [B2, T, N, C1]
        mc_flag = question['mc_flag']  # [B2n]

        # 把一个 batch 中的 B 条视频，根据 mc_flag 复制 / 重排成 Bn 条视频，使它们与 Bn 条多选问题一一对应
        v_embedding = v_embedding[mc_flag.long()]  # [B2n, T, N, C1]
        B, T, N, _ = v_embedding.shape

        # need to split question and choice text
        q_tokens, q_pad_mask = question['mc_q_tokens'], question['mc_q_pad_mask']
        q_embedding = self.q_embedding(q_tokens)  # [B2n, L, C2]
        question = q_embedding[:, :self.mc_question_len]

        choice = q_embedding[:, self.mc_question_len:]
        q_embedding = pt.cat([
            self.batch_cat_vec(question, self.mc_question_token, dim=-1),
            self.batch_cat_vec(choice, self.mc_choice_token, dim=-1),
        ], 1)  # [B2n, L, C2+2]

        in_embedding, pad_mask, mask_idx, gt_v_emb = self._process_in_embeddings(v_embedding, q_embedding, q_pad_mask)

        in_embedding = self.posit_embed(in_embedding)
        transformer_out = self.backbone(in_embedding, src_key_padding_mask=pad_mask)  # [B2n, 1 + T*N + L, d_model]

        # binary classification
        cls_emb = transformer_out[:, 0, :]  # [B2n, d_model]
        answer_logits = self.mlp_a_mc(cls_emb)  # [B2n, 1],

        # masked object prediction
        if self.mask_obj_loss:
            out_v_emb = transformer_out[:, 1:1 + T * N].reshape(B * T, N, -1)  # 该 out_v_emb 来自已被 mask 的 in_embedding，但这里它已经将 “被 mask 区域” 通过 self—attention 补齐了
            batch_idx = pt.arange(B * T, device=out_v_emb.device)
            mask_v_emb = out_v_emb[batch_idx, mask_idx]  # [B*T, C], 提取 self—attention 补齐后的 mask 区域
            pred_v_emb = self.mask_obj_fc(mask_v_emb)  # [B * T, C1]
        else:
            pred_v_emb = None

        return {
            'answer_logits': answer_logits.view(-1),
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }


    def forward(self, video_emb, question):
        """
        Args:
            video_emb: [b, t, n, c]

            question:{
                    - question_type: Tensor, (b,)

                    - cls_q_tokens: Tensor, (b1, L,)
                    - cls_q_pad_mask: Tensor, (b1, L,)
                    - cls_label: Tensor, (b1,)

                    - mc_subtype: Tensor, (b2, )
                    - mc_q_tokens: Tensor, (b2n, L), concated along num_choices dim
                    - mc_q_pad_mask: Tensor, (b2n, L)
                    - mc_label: Tensor, (b2n,)
                    - mc_flag: Tensor, (b2n,) e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, ...]
                }

        Returns:
            torch.Tensor: [B, num_cls/num_choices], predicted answer logits
        """
        cls_dict = self._cls_forward(video_emb, question)  # 处理分类问题
        mc_dict = self._mc_forward(video_emb, question)  # 处理多选问题

        cls_answer_logits = cls_dict['answer_logits'] if cls_dict is not None else None
        mc_answer_logits = mc_dict['answer_logits'] if mc_dict is not None else None

        if self.mask_obj_loss:
            gt_v_emb = pt.cat([d['gt_v_emb'] for d in [cls_dict, mc_dict] if d is not None],0)  # true v_mask_emb
            pred_v_emb = pt.cat([d['pred_v_emb'] for d in [cls_dict, mc_dict] if d is not None], 0)  # pred v_mask_emb
        else:
            gt_v_emb, pred_v_emb = None, None

        return {
            'cls_answer_logits': cls_answer_logits,  # [B1, num_cls]
            'mc_answer_logits': mc_answer_logits,  # [B2n]
            'gt_v_emb': gt_v_emb,
            'pred_v_emb': pred_v_emb,
        }


    @staticmethod
    def batch_cat_vec(tensor, value_vec, dim):
        """Concat some values at the end of a tensor along one dim.

        Useful in e.g. concat some indicator to different input data.

        Args:
            tensor (torch.Tensor): [N_1, N_2, ..., N_n].
            value_vec (torch.Tensor | List[Any]): a d-len vector to be concated.
            dim (int): specifies the dimention.

        Returns:
            torch.Tensor: [N_1, N_2, ... N_{dim} + d, ..., N_n].
        """
        assert isinstance(tensor, pt.Tensor)
        assert isinstance(dim, int)
        tensor_shape = list(tensor.shape)
        n = len(tensor_shape)
        if dim < 0:
            dim = n + dim
        if not isinstance(value_vec, pt.Tensor):
            value_vec = pt.tensor(value_vec)
        value_vec = value_vec.type_as(tensor)
        # expand shape to match tensor
        for dim_ in range(n):
            if dim_ != dim:
                value_vec = value_vec.unsqueeze(dim=dim_)
        tensor_shape[-1] = 1
        value_vec = value_vec.repeat(tensor_shape)
        return pt.cat([tensor, value_vec], dim=dim)




