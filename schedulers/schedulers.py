"""
===================================
░▒▓ T E T R A C H R O M A T I C ▓▒░
===================================

A simple collection of stateless learning rate schedulers.

I developed this file because I found the PyTorch schedulers to
be overly complex for my purposes. If we are willing to enforce
statelessness, then we can express any schduler as a pure function
mapping the current step to the current lr. For convenience, I
also provide the option to pass in a PyTorch optimiser object and
set its lr as a side effect.

To emphasise simplicity and statelessness, the schedulers are implemented
as frozen, callable, dataclasses. While they take arguments at
initialisation-time, these just serve to fix the function parameters.
I.e.
> scheduler = ParametricScheduler(parameter=x)
is conceptually equivalent to
> scheduler = partial(parametric_scheduler_fn, parameter=x)

When using with PyTorch, these schedulers do not need to be
saved in checkpoints or moved to GPU. I've used them in up to 32-GPU
distributed training (DDP) setups and they have never been a bottleneck.

To use, just copy-paste this file into your project. Feel free to
delete this docstring -- the code is yours now!

(although I do ask you to please keep the license comment below <3)
"""

# Forked from github.com/Charl-AI/tetrachromatic under MIT license.

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _set_lr(optimizer: torch.optim.Optimizer | None, lr: float) -> None:
    if optimizer is None:
        return
    for pg in optimizer.param_groups:
        pg["lr"] = lr  # in-place update


@dataclass(frozen=True)
class Scheduler(ABC):
    @abstractmethod
    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Calculates the learning rate from the current step.
        Returns the new learning rate. If an optimiser is given,
        its learning rate will be set to this value as a side effect.

        Note that this will override the learning rate that you
        initially instantiated the optimizer with. It will also
        set a homogeneous lr for all param_groups, even if you initially
        created the optmizer with separate lrs for different groups.

        Usage (should be called on each training step):
        ```python
        loss = ...
        loss.backward()
        lr = scheduler(global_step, optimizer)
        optimizer.step()
        optimizer.zero_grad()
        ```
        """
        pass


@dataclass(frozen=True)
class ConstantLR(Scheduler):
    """Constant learning rate. Mostly useful as a dummy scheduler."""

    lr: float

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        del global_step
        _set_lr(optimizer, self.lr)
        return self.lr


@dataclass(frozen=True)
class LinearLR(Scheduler):
    """Linear warmup followed by linear decay (triangular shape).

    Bergsma et al. [1] find that a linear schedule with min_lr=0
    works particularly well for training LLMs with AdamW.

    [1] https://arxiv.org/abs/2502.15938
    """

    warmup_steps: int
    total_steps: int
    min_lr: float
    max_lr: float

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        new_lr: float
        decay_steps = max(1, self.total_steps - self.warmup_steps)

        if global_step <= self.warmup_steps:  # linear warmup
            new_lr = self.min_lr + global_step / self.warmup_steps * (
                self.max_lr - self.min_lr
            )
        else:
            steps_into_decay = global_step - self.warmup_steps
            decay_progress = min(1.0, steps_into_decay / decay_steps)
            new_lr = self.max_lr + decay_progress * (self.min_lr - self.max_lr)

        # clip to min/max
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        _set_lr(optimizer, new_lr)
        return new_lr


@dataclass(frozen=True)
class CosineLR(Scheduler):
    """Cosine learning rate scheduler with linear warmup.
    For LLMs, it's common practice to set min_lr = 0.1 * max_lr.
    See Chinchilla [1] for arguably the best known usage.

    [1] https://arxiv.org/abs/2203.15556
    """

    # While the code has been extensively changed, note that this was orignially
    # forked from github.com/apple/ml-tarflow under Apple open source license:
    # https://github.com/apple/ml-tarflow/blob/main/LICENSE

    warmup_steps: int
    total_steps: int
    min_lr: float
    max_lr: float

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        if global_step <= self.warmup_steps:  # linear warmup
            new_lr = self.min_lr + global_step / self.warmup_steps * (
                self.max_lr - self.min_lr
            )
        else:  # cosine decay
            t = (global_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (
                self.max_lr - self.min_lr
            )

        # clip to min/max
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        _set_lr(optimizer, new_lr)
        return new_lr


@dataclass(frozen=True)
class InvSqrtLR(Scheduler):
    """Inverse square-root decay scheduler.
    This scheduler is particularly useful for training diffusion
    models and is designed to keep the parameter norms constant
    throughout training. See Fig. 3 in Karras et al. [1].

    This scheduler is composed of three phases:
      1. Linear warmup from 0 to max_lr over `warmup_steps` duration.
      2. Constant at max_lr for `constant_steps` duration.
      3. Inverse sqrt decay (can continue essentially indefinitely).

    (if constant_steps = 0, there is no phase 2)

    The decay factor in phase 3 is:
    sqrt(global_step / (warmup_steps + constant_steps)).

    Thus, if you want to target a final lr of 0.1 x max_lr,
    you should set warmup_steps + constant_steps to 1% of training.

    [1] https://arxiv.org/abs/2312.02696
    """

    # Karras et al. use a different notation and units, but their
    # settings for EDM2-M-ImageNet64 are approximately equivalent to:
    #   - warmup_steps ~5k
    #   - constant_steps ~30k
    #   - max_lr 0.009
    # They train for ~1M steps, resulting in a final lr of ~0.0017
    # and final decay factor of ~5x

    warmup_steps: int
    constant_steps: int
    max_lr: float

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        decay_start = self.warmup_steps + self.constant_steps
        new_lr = self.max_lr

        new_lr *= min(global_step / self.warmup_steps, 1.0)  # apply warmup
        new_lr /= math.sqrt(max(global_step / decay_start, 1.0))  # apply decay

        _set_lr(optimizer, new_lr)
        return new_lr
