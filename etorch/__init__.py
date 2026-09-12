"""Backward-compatibility alias: ``etorch`` was renamed to ``storch``.

Please update your imports::

    import storch
    import storch.nn as nn

This shim will be removed in a future release.
"""
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "The 'etorch' package was renamed to 'storch'. "
    "Update your imports (import storch). This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

import storch as _storch

_sys.modules[__name__] = _storch
