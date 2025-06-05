import time

import pytest
import torch

from tests.utils.mnist import MnistModel, MnistModelConfig
from trainer import Trainer, TrainerArgs
from trainer.generic_utils import KeepAverage
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

    model = MnistModel()
    # Test StepwiseGradualLR
    config = MnistModelConfig(
        lr_scheduler="StepwiseGradualLR",
        lr_scheduler_params={
            "gradual_learning_rates": [
                [0, LR_1],
                [2, LR_2],
            ]
        },
        scheduler_after_epoch=False,
    )
    trainer = Trainer(TrainerArgs(), config, output_path=tmp_path, model=model, gpu=0 if is_cuda else None)
    trainer.train_loader = trainer.get_train_dataloader(
        trainer.training_assets,
        trainer.train_samples,
        verbose=True,
    )
    trainer.keep_avg_train = KeepAverage()

    lr_0 = trainer.scheduler.get_lr()
    trainer.train_step(next(iter(trainer.train_loader)), len(trainer.train_loader), 0, time.time())
    lr_1 = trainer.scheduler.get_lr()
    trainer.train_step(next(iter(trainer.train_loader)), len(trainer.train_loader), 1, time.time())
    lr_2 = trainer.scheduler.get_lr()
    assert lr_0 == [LR_1]
    assert lr_1 == [LR_1]
    assert lr_2 == [LR_2]
