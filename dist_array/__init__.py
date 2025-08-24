from importlib.metadata import version
import logging

from .core import serialize
from .core import protocol

__version__ = version("dist_array")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")