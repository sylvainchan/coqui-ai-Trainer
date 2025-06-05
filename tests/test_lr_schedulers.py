import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig, create_trainer, run_steps
from trainer.torch import StepwiseGradualLR

is_cuda = torch.cuda.is_available()


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(2, 1)
    return torch.optim.SGD(model.parameters(), lr=0.1)


@pytest.fixture
def lr_schedule():
    return [
        (0, 0.01),  # initial learning rate
        (5, 0.02),  # increase at epoch 5
        (10, 0.05),  # increase at epoch 10
        (20, 0.1),  # final level
    ]


@pytest.mark.parametrize(
    "last_epoch, expected_lr",
    [
        (1, 0.01),  # Before second threshold
        (5, 0.02),  # Exact match with second threshold
        (7, 0.02),  # Between 5 and 10
        (10, 0.05),  # Exact match with third threshold
        (11, 0.05),  # Between 10 and 20
        (25, 0.1),  # After last threshold
    ],
)
def test_stepwise_lr_schedule(dummy_optimizer, lr_schedule, last_epoch, expected_lr):
    scheduler = StepwiseGradualLR(dummy_optimizer, lr_schedule)
    scheduler.last_epoch = last_epoch
    lrs = scheduler.get_lr()
    assert lrs == [expected_lr] * len(dummy_optimizer.param_groups)


def test_train_mnist(tmp_path):
    LR_1 = 1e-3
    LR_2 = 1e-4

    config = MnistModelConfig(
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={"gradual_learning_rates": [(0, LR_1), (2, LR_2)]},
        scheduler_after_epoch=False,
    )
    trainer = create_trainer(config, MnistModel(), tmp_path, gpu=0 if is_cuda else None)

    assert trainer.scheduler.get_lr() == [LR_1]

    run_steps(trainer, 0, 1)
    assert trainer.scheduler.get_lr() == [LR_1]

    run_steps(trainer, 1, 2)
    assert trainer.scheduler.get_lr() == [LR_2]
