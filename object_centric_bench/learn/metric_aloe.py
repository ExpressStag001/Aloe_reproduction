import torch as pt

from .metric import Metric

class GaussianVarianceKLDLoss(Metric):
    def __init__(self, dim, log_variance, mean=()):
        super().__init__(mean)
        self.dim = dim
        self.log_variance = log_variance

    def forward(self, input):
        """
        KLD between N(mu, sigma^2) and N(mu, sigma0^2) (variance regularization).
        and where sigma0^2 = exp(self.log_variance) is fixed
        """

        assert input.shape[-1] == self.dim * 2
        mu1 = input[..., :self.dim]
        log_var1 = input[..., self.dim:]
        mu2 = mu1.detach().clone()  # no penalty for mu
        log_var2 = pt.ones_like(log_var1).detach() * self.log_variance
        sigma1 = pt.exp(log_var1 * 0.5)
        sigma2 = pt.exp(log_var2 * 0.5)
        kld = (pt.log(sigma2 / sigma1) + (pt.exp(log_var1) + (mu1 - mu2)**2) / (2. * pt.exp(log_var2)) - 0.5).sum(-1)  # (b,), 一维高斯分布下的 KL 散度的公式
        kld= kld.mean()[None]  # (b=1,)

        return self.finaliz(kld)  # (b,) (b,)


class ClevrerVQAAccuracy(Metric):
    """"""

    def __init__(self, q_type, subtype=None, mean=()):
        super().__init__(mean)
        assert q_type in [0,1]
        assert subtype in [1,2,3,None]
        self.q_type = q_type
        self.subtype = subtype

    def forward(self, input, target, mc_flag=None, mc_subtype=None):
        '''
        input:
            cls_answer_logits = (b1, num_cls) / mc_answer_logits = (b2n,)

        target:
            cls_labels = (b1,) /  mc_label: Tensor, (b2n,)

        mc_flag:
             mapping choice to question , (b2n,) e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2],

        mc_subtype:
            (b2,)
        '''

        if self.q_type ==0:  # cls
            acc = (input.argmax(-1) == target).float()  # (b1,)

        else:  # mc
            assert mc_flag is not None # only for mc_question
            mc_preds = (input > 0.).type_as(target)
            mc_correct_mask = (mc_preds == target).float()  # e.g. [1, 1, 1, 1 ,0, 1, 1, 0, 0]
            mc_q_num = mc_flag.max().item() + 1  # b2
            mc_corr_ques = []
            for i in range(mc_q_num):
                mc_corr_ques.append(mc_correct_mask[mc_flag == i].all())  # all() 表示只有当一个问题的 choice 全部正确时，该 mc_question 才正确
            mc_corr_ques = pt.stack(mc_corr_ques).cuda()  # (b2,)

            if self.subtype is None:  # mc_acc
                acc = mc_corr_ques
            else:
                assert mc_subtype is not None
                subtype_mask = (mc_subtype == self.subtype)
                if not subtype_mask.any():
                    return self.finaliz(pt.zeros(1).cuda(), pt.zeros(1, dtype=pt.bool).cuda())
                acc = mc_corr_ques[subtype_mask]

        return self.finaliz(acc)