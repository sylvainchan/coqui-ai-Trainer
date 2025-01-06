import pytest

from trainer.config import TrainerConfig


def test_optimizer_params():
    TrainerConfig(
        optimizer="optimizer",
        grad_clip=0.0,
        lr=0.1,
        optimizer_params={},
        lr_scheduler="scheduler",
    )

    TrainerConfig(
        optimizer=["optimizer1", "optimizer2"],
        grad_clip=[0.0, 0.0],
        lr=[0.1, 0.01],
        optimizer_params=[{}, {}],
        lr_scheduler=["scheduler1", "scheduler2"],
    )

    with pytest.raises(TypeError, match="Either none or all of these fields must be a list:"):
        TrainerConfig(
            optimizer=["optimizer1", "optimizer2"],
            grad_clip=0.0,
            lr=[0.1, 0.01],
            optimizer_params=[{}, {}],
            lr_scheduler=["scheduler1", "scheduler2"],
        )
