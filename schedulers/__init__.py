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

---

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

---

How to choose a scheduler:

- if you are using a pretrain-finetune strategy, use ConstantLR for
  pretraining and CooldownLR for finetuning.
- if you know the amount of steps you want to train for in advance,
  CosineLR is generally sensible.
- if you don't want to commit to a number of steps in advance and
  want a reasonable and somewhat principled default, use InvSqrtL.

In my experience (mostly training 1B+ param vision diffusion models),
these rules of thumb should serve you fairly well. I don't have
much experience on LLMs, so YMMV.

---

To use, just copy-paste this file into your project. Feel free to
delete this docstring -- the code is yours now!

(although I do ask you to please keep the license comment below <3)
"""

# Forked from github.com/Charl-AI/tetrachromatic under MIT license.

from __future__ import annotations

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
        # avoids recompilation when using torch.compile
        if not isinstance(pg["lr"], torch.Tensor):
            pg["lr"] = torch.tensor(pg["lr"])
        pg["lr"].fill_(lr)  # in-place update


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
    """Constant learning rate with optional warmup from min_lr to max_lr."""

    max_lr: float
    min_lr: float = 0
    warmup_steps: int = 0

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        if self.warmup_steps != 0 and global_step <= self.warmup_steps:
            t = global_step / self.warmup_steps
            new_lr = t * self.max_lr + (1 - t) * self.min_lr
        else:
            new_lr = self.max_lr

        new_lr = max(self.min_lr, min(self.max_lr, new_lr))  # clip if needed
        _set_lr(optimizer, new_lr)
        return new_lr


@dataclass(frozen=True)
class CooldownLR(Scheduler):
    """Linearly decay from max_lr to min_lr over cooldown_steps.
    This is generally useful for finetuning at the end of long,
    constant lr training runs (see Hägele et al. [1]).
    [1] https://arxiv.org/abs/2405.18392v3.
    """

    max_lr: float
    min_lr: float
    cooldown_steps: int

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        t = global_step / self.cooldown_steps
        new_lr = t * self.min_lr + (1 - t) * self.max_lr

        new_lr = max(self.min_lr, min(self.max_lr, new_lr))  # clip if needed
        _set_lr(optimizer, new_lr)
        return new_lr


@dataclass(frozen=True)
class CosineLR(Scheduler):
    """Cosine learning rate scheduler with optional linear warmup.
    For LLMs, it's common practice to set min_lr = 0.1 * max_lr.
    See Chinchilla [1] for arguably the best known usage.

    [1] https://arxiv.org/abs/2203.15556
    """

    # While the code has been extensively changed, note that this was orignially
    # forked from github.com/apple/ml-tarflow under Apple open source license:
    # https://github.com/apple/ml-tarflow/blob/main/LICENSE

    max_lr: float
    min_lr: float
    total_steps: int
    warmup_steps: int = 0

    def __call__(
        self,
        global_step: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        if self.warmup_steps != 0 and global_step <= self.warmup_steps:
            new_lr = self.min_lr + global_step / self.warmup_steps * (
                self.max_lr - self.min_lr
            )
        else:
            t = (global_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (
                self.max_lr - self.min_lr
            )

        new_lr = max(self.min_lr, min(self.max_lr, new_lr))  # clip if needed
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

    max_lr: float
    warmup_steps: int
    constant_steps: int

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
