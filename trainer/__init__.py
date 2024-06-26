import os

from trainer.model import *
from trainer.trainer import *

with open(os.path.join(os.path.dirname(__file__), "VERSION"), encoding="utf-8") as f:
    version = f.read().strip()

__version__ = version
