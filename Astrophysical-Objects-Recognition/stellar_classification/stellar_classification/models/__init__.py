"""Models sub-package."""

from .network import SimpleNN  # noqa: F401
from .trees import SimpleRandomForest, SimpleExtraTrees #aggiunta enrica 

__all__ = ['SimpleNN', 'SimpleRandomForest', 'SimpleExtraTrees'] #aggiunti i modelli da enrica 