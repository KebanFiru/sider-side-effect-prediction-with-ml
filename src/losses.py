"""Loss functions and helpers for GIN-v2 ablations 2.2 (asymmetric loss) and
2.5 (task weighting). With the default configuration (ablation.use_asl=false,
ablation.task_weighting=none) none of this is active: the baseline (focal loss,
equal task weights) behaves exactly as in `src/train.py`.
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021 (https://github.com/Alibaba-MIIL/ASL).

    A negative label in SIDER means "not listed on the label", not "genuinely
    absent", which makes the data positive-unlabelled. ASL does not model that
    directly, but by focusing separately on the positive and negative classes
    (gamma_neg != gamma_pos) and shifting the negative probability (clip), it
    reduces the dominance of easy negatives -- some of which, under a
    positive-unlabelled regime, may really be mislabelled positives.

    L = -y * (1-p)^gamma_pos * log(p)
        - (1-y) * (p_m)^gamma_neg * log(1-p_m),   p_m = max(p - clip, 0)

    `gamma_neg > gamma_pos` (default 4 vs 1) suppresses the negative class more
    aggressively, since negatives are the overwhelming majority in SIDER.
    `clip` (default 0.05) shifts the probability enough to drop very easy
    negatives (where p is already tiny) from the loss entirely.

    Note: `src.train.bake_in_task_alpha_threshold`, which folds focal loss's
    per-task alpha into the final layer's bias, must not be used together with
    this loss. ASL's own asymmetry already does that job, and applying both
    shifts the threshold twice. `src/train.py::build_criterion` skips it
    automatically (see the `use_asl` check there).
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05,
                 eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps)) * (1 - p_pos).clamp(min=0) ** self.gamma_pos
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps)) * p_neg ** self.gamma_neg
        return -(loss_pos + loss_neg)


class UncertaintyWeighting(nn.Module):
    """Kendall et al. 2018 (multi-task uncertainty weighting).

    Holds a learnable log(sigma_t) per task. The total loss is
    sum_t [ task_loss_t / (2*sigma_t^2) + log(sigma_t) ], so the model can grow
    the sigma of a noisy or hard task and reduce that task's gradient
    contribution by itself, unlike fixed equal weighting.

    This module's parameters must also be added to the optimizer (see
    `src/train.py::build_optimizer_and_scheduler`), otherwise sigma is never
    learned and the effect is nil.
    """

    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, per_task_loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`per_task_loss`: (num_tasks,) -- each task's mean loss, already
        weighted by the mask. `valid`: (num_tasks,) bool, the tasks that have
        labels in this batch. Returns a single scalar loss.

        Only `valid` tasks are counted: for a task with no labels in the batch
        the loss is 0 but the gradient of the `+log_sigma` term is still 1, so
        that task's sigma would shrink on every empty batch and its weight
        would balloon once the task reappeared.

        The mean is taken instead of the sum: each task's optimal sigma is
        unchanged, but the loss keeps a scale close to the baseline's (summing
        would make it about 60 times larger and shrink the relative effect of
        Adam's weight decay).
        """
        precision = torch.exp(-2 * self.log_sigma)
        return (precision * per_task_loss + self.log_sigma)[valid].mean()


def per_task_masked_loss(elementwise_loss: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
    """`elementwise_loss`, `weight_mask`: (batch, num_tasks). Returns
    (num_tasks,): each task's mean loss over the samples labelled for that task
    (weight_mask>0). A task with no labelled sample returns 0, which
    contributes nothing to the gradient and produces no NaN.
    """
    num = (elementwise_loss * weight_mask).sum(dim=0)
    den = weight_mask.sum(dim=0).clamp(min=1e-8)
    per_task = num / den
    return torch.where(weight_mask.sum(dim=0) > 0, per_task, torch.zeros_like(per_task))
