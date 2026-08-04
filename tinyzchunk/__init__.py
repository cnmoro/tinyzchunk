"""tinyzchunk: GPU-free chunking distilled from the zChunk algorithm."""

from .chunker import Chunker, chunk, get_chunker  # noqa: F401
from .features import extract_features  # noqa: F401
from .model import TinyChunkModel  # noqa: F401

__version__ = "0.3.0"
