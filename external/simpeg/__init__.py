# Bridge: redirect to the real simpeg package at simpeg/simpeg/
#
# The editable install maps 'simpeg' -> 'simpeg/simpeg/', but the outer
# simpeg/ directory shadows it as a namespace package when running from
# the project root.  This __init__.py fixes the resolution.
#
# Two-step approach:
# 1. Load the inner simpeg.simpeg package and copy all its exports.
# 2. Register copied submodules under their simpeg.X names in sys.modules
#    so that later imports like "from simpeg.data import X" find the cached
#    modules instead of re-importing them via __path__ (which would create
#    duplicate class objects and break isinstance checks).

import sys as _sys
import types as _types

# Step 1: import the real package (found via standard subpackage lookup from
# the outer simpeg/ directory)
from .simpeg import *  # noqa: F401, F403, E402
from .simpeg import __version__  # noqa: F401

# Step 2: register submodules under their top-level simpeg.X names so that
# "import simpeg.data" hits sys.modules['simpeg.data'].
_simpeg_inner = _sys.modules['simpeg.simpeg']
for _attr_name in dir(_simpeg_inner):
    if _attr_name.startswith('_'):
        continue
    _attr_value = getattr(_simpeg_inner, _attr_name)
    if isinstance(_attr_value, _types.ModuleType):
        _alias = f'simpeg.{_attr_name}'
        _sys.modules.setdefault(_alias, _attr_value)

del _sys, _types, _simpeg_inner, _attr_name, _attr_value, _alias
