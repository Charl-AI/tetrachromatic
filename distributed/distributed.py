"""
===================================
░▒▓ T E T R A C H R O M A T I C ▓▒░
===================================

Distributed training utilities for PyTorch.

The core object in this file is the `Distributed` class.
When instantiated, it sets up the configuration necessary
for distributed training. It also contains a core set of
convenience methods and properties (e.g. getting local
rank and device, all_gather, etc.).

We assume you are using submitit to launch your jobs.
If you would like to use another option (e.g. torchrun),
you can simply re-write the __init__ method to support it.

The Distributed object is a singleton. Once it's been
instantiated, any further calls will return the initial instance:
```
D1 = Distributed() # calls __init__ and sets up distributed training
D2 = Distributed() # skips __init__ and returns the initial instance
assert id(D1) == id(D2)
```

The singleton style enables two distinct usage patterns:

1. Dependency injection pattern.
```python
def my_func(arg1, arg2, D: Distributed):
    rank = D.rank
    world_size = D.world_size
    print(f"{rank=}, {world_size=}")
    ... # do something with arg1, arg2

D = Distributed()  # setup
my_func(arg1, arg2, D)
```

2. Global variable pattern.
```python
def my_func(arg1, arg2):
    rank = Distributed().rank
    world_size = Distributed().world_size
    print(f"{rank=}, {world_size=}")
    ... # do something with arg1, arg2

Distributed()  # setup (not strictly needed)
my_func(arg1, arg2)

```

The dependency injection pattern is more explicit, but
requires you to do the work of 'plumbing' the instance
to every function where it is needed. The global variable
approach avoids this work, but makes it harder to inspect
functions that depend on it.

To try and get the best of both worlds, I've included an
optional experimental decorator that can be used to inject
the object as an argument to functions without plumbing
it throughout your code:

3. Decorator.
```python
@distributed
def my_func(arg1, arg2, *, D: Distributed):
    rank = D.rank
    world_size = D.world_size
    print(f"{rank=}, {world_size=}")
    ... # do something with arg1, arg2

D = Distributed()        # setup (not strictly needed)
my_func(arg1, arg2)      # no need to pass D to my_func
my_func(arg1, arg2, D=D) # but equally, no problem if you decide to

```
---

As well as the Distributed object, we include a decorator
that can be used to only run a function on the rank zero
process. This can be used as a standalone utility.

---

To use, just copy-paste this file into your project. Feel free to
delete this docstring -- the code is yours now!

(although I do ask you to please keep the license comment below <3)
"""

# Forked from github.com/Charl-AI/tetrachromatic under MIT license.

import builtins
import functools
import inspect
import os
from typing import Literal

import torch
import torch.distributed as dist
from submitit.helpers import TorchDistributedEnvironment


def singleton(cls):
    """Decorate a class to turn it into a singleton."""
    instances = {}

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class Distributed:
    def __init__(self):
        print = functools.partial(builtins.print, flush=True)
        dist_env = TorchDistributedEnvironment()
        dist_env.export(set_cuda_visible_devices=True, overwrite=True)

        print("Setting up Distributed environment")
        print(f"Master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"Rank: {dist_env.rank}")
        print(f"Local rank: {dist_env.local_rank}")
        print(f"Visible device ID: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"World size: {dist_env.world_size}")
        print(f"Local world size: {dist_env.local_world_size}")

        dist.init_process_group(
            backend="nccl",
            world_size=dist_env.world_size,
            rank=dist_env.rank,
        )
        assert dist_env.rank == dist.get_rank()
        assert dist_env.world_size == dist.get_world_size()

        # assume one process per GPU, where each process gets the GPU index
        # corresponding to local_rank
        assert dist_env.local_rank == int(os.environ.get("CUDA_VISIBLE_DEVICES"))  # type:ignore

        self._rank = dist_env.rank
        self._local_rank = dist_env.local_rank
        self._world_size = dist_env.world_size
        self._local_world_size = dist_env.local_world_size
        self._device = torch.device(f"cuda:{self._local_rank}")
        torch.cuda.set_device(self._device)
        self.barrier()

    @property
    def world_size(self) -> int:
        """The number of processes participating in the job.
        Usually equal to num_nodes * num_gpus per node."""
        return self._world_size

    @property
    def rank(self) -> int:
        """The rank of the current process. Takes a range of [0, world_size).
        It is conventional to treat rank 0 as the master process."""
        return self._rank

    @property
    def device(self) -> torch.device:
        """The GPU device associated with the current process."""
        return self._device

    def barrier(self):
        """Synchronise all processes."""
        if dist.is_initialized():
            dist.barrier()

    def gather_concat(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Gather a tensor from all processes and concat along a given dimension.

        Example:
        ```python
        D = Distributed()
        outputs = model(x)  # shape (B, C, H, W)
        outputs = D.gather_concat(outputs)  # shape (world_size * B, C, H, W)
        ```
        """
        if dist.is_initialized():
            x_list = [torch.empty_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(x_list, x)
            return torch.cat(x_list, dim=dim)
        else:
            return x

    def all_reduce(
        self,
        x: torch.Tensor,
        op: Literal["sum", "avg", "product", "min", "max"] = "sum",
    ) -> torch.Tensor:
        """Reduce a tensor by summing or averaging across all processes.

        Example:
        ```python
        D = Distributed()
        x = torch.randn(10, device=D.device) + 1  # shape (10,), mu=1, var=1 on each process
        y = D.all_reduce(x, op="sum")  # shape (10,), mu=world_size, var=world_size
        z = D.all_reduce(x, op="avg")  # shape (10,), mu=1, var=(1 / world_size)
        ```
        """
        if dist.is_initialized():
            x = x.clone()
            match op:
                case "sum":
                    reduce_op = dist.ReduceOp.SUM
                case "avg":
                    reduce_op = dist.ReduceOp.AVG
                case "product":
                    reduce_op = dist.ReduceOp.PRODUCT
                case "min":
                    reduce_op = dist.ReduceOp.MIN
                case "max":
                    reduce_op = dist.ReduceOp.MAX
                case _:
                    raise ValueError(f"Invalid reduction operation: {op}")
            dist.all_reduce(x, op=reduce_op)
            return x
        else:
            return x

    def __del__(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def distributed(func):
    """Decorate a function to inject the `Distributed` singleton.
    Requires you to reserve `D` as a keyword-only argument.

    Usage:
    ```python
    @distributed
    def my_func(arg1, arg2, *, D: Distributed):
        # you can use D like normal!
        ...

    my_func(arg1, arg2)  # no need to pass D here
    ```
    """

    # you may be wondering why we can't simply provide D=Distributed()
    # as a default argument to func. The issue with this is that Distributed()
    # would be called at function creation time, not when the function is called.
    # This would cause us to lose control of when to initialise the process group.
    # For example, when using submitit, you don't want to call Distributed()
    # until you are inside the running job (i.e. in the submitted callable).

    signature = inspect.signature(func)
    assert "D" in signature.parameters
    assert signature.parameters["D"].kind == inspect.Parameter.KEYWORD_ONLY

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "D" not in kwargs:
            kwargs["D"] = Distributed()
        return func(*args, **kwargs)

    return wrapper


def rank_zero(barrier: bool = False):
    """Decorate a function to only call it on rank zero processes.
    Can optionally force sync afterwards with `dist.barrier()`.

    The decorated function returns None on all non-zero processes.
    It is thus best to use this decorator with side-effect functions
    like saving and logging where a return value is not needed.

    Usage:
    ```python
    @rank_zero()
    def my_print(msg):
        print(msg)

    @rank_zero(barrier=True)
    def write_to_disk(filename, data):
        with open(filename, "w") as f:
            f.write(data)
        time.sleep(1) # Simulate a slow I/O operation
    ```
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            is_dist = dist.is_available() and dist.is_initialized()
            rank0 = not is_dist or dist.get_rank() == 0
            if rank0:
                result = func(*args, **kwargs)
            if barrier and is_dist:
                dist.barrier()
            return result

        return wrapper

    return decorator
