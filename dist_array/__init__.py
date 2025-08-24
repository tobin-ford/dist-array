from importlib.metadata import version
import logging

from .core import serialize

__version__ = version("dist_array")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel("DEBUG")