import matplotlib.pyplot as plt
import torch

import schedulers as sch

torch._logging.set_logs(recompiles=True)  # type: ignore


def test_recompile():
    model = torch.nn.Sequential(
        *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
    )
    input = torch.rand(1024, device="cuda")
    output = model(input)
    output.sum().backward()

    opt = torch.optim.Adam(model.parameters())
    s = sch.InvSqrtLR(warmup_steps=5_000, constant_steps=45_000, max_lr=1e-4)

    @torch.compile(fullgraph=False)
    def fn(step, opt, s):
        lr = s(step, opt)
        opt.step()
        return lr

    # Warmup runs to compile the function
    for step in range(5):
        fn(step, opt, s)
        print(opt.param_groups[0]["lr"])


def test_visualise():
    s1 = sch.ConstantLR(max_lr=1e-4, min_lr=1e-6, warmup_steps=10_000)
    s2 = sch.CooldownLR(max_lr=1e-4, min_lr=0, cooldown_steps=900_000)
    s3 = sch.CosineLR(
        max_lr=1e-4,
        min_lr=1e-5,
        warmup_steps=4_000,
        total_steps=1_000_000,
    )
    s4 = sch.InvSqrtLR(warmup_steps=5_000, constant_steps=45_000, max_lr=1e-4)

    steps = list(range(1_000_000))
    lrs = []

    for s in [s1, s2, s3, s4]:
        lrs = []
        for step in steps:
            lrs.append(s(global_step=step))
        plt.plot(steps, lrs, label=f"{s.__class__}")

    plt.legend()
    plt.show()
