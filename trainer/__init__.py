import importlib.metadata

from trainer.config import TrainerArgs, TrainerConfig
from trainer.model import *
from trainer.trainer import *

__version__ = importlib.metadata.version("coqui-tts-trainer")
