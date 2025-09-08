"""
IK Adapters Module - Unified interface for different IK solvers.

This module provides adapters to integrate various IK solvers (pyroki, curobo) 
into the HIROL platform's benchmark system while maintaining consistent interfaces.

Uses lazy loading to avoid importing heavy dependencies until actually needed.
"""

from .base_adapter import IKAdapterBase

# Lazy loading globals - not imported until needed
_PyrokiAdapter = None
_CuroboAdapter = None
_PYROKI_CHECKED = False
_CUROBO_CHECKED = False
_PYROKI_AVAILABLE = False
_CUROBO_AVAILABLE = False


def _check_pyroki():
    """Check if Pyroki adapter is available (lazy check)."""
    global _PyrokiAdapter, _PYROKI_CHECKED, _PYROKI_AVAILABLE
    if not _PYROKI_CHECKED:
        try:
            from .pyroki_adapter import PyrokiAdapter
            _PyrokiAdapter = PyrokiAdapter
            _PYROKI_AVAILABLE = True
        except ImportError:
            _PyrokiAdapter = None
            _PYROKI_AVAILABLE = False
        _PYROKI_CHECKED = True
    return _PYROKI_AVAILABLE


def _check_curobo():
    """Check if CuRobo adapter is available (lazy check)."""
    global _CuroboAdapter, _CUROBO_CHECKED, _CUROBO_AVAILABLE
    if not _CUROBO_CHECKED:
        try:
            from .curobo_adapter import CuroboAdapter
            _CuroboAdapter = CuroboAdapter
            _CUROBO_AVAILABLE = True
        except ImportError:
            _CuroboAdapter = None
            _CUROBO_AVAILABLE = False
        _CUROBO_CHECKED = True
    return _CUROBO_AVAILABLE


__all__ = [
    'IKAdapterBase',
    'PyrokiAdapter',
    'CuroboAdapter', 
    'PYROKI_AVAILABLE',
    'CUROBO_AVAILABLE'
]


def __getattr__(name):
    """Module-level lazy loading using __getattr__."""
    if name == 'PyrokiAdapter':
        if _check_pyroki():
            return _PyrokiAdapter
        return None
    elif name == 'CuroboAdapter':
        if _check_curobo():
            return _CuroboAdapter
        return None
    elif name == 'PYROKI_AVAILABLE':
        return _check_pyroki()
    elif name == 'CUROBO_AVAILABLE':
        return _check_curobo()
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


def get_available_adapters():
    """Get list of available IK adapters."""
    available = []
    if _check_pyroki():
        available.append('pyroki')
    if _check_curobo():
        available.append('curobo')
    return available


def create_adapter(adapter_type: str, urdf_path: str, end_effector_link: str, **kwargs):
    """Factory function to create IK adapters."""
    if adapter_type.lower() == 'pyroki':
        if not _check_pyroki():
            raise ImportError("pyroki adapter not available. Please install pyroki dependencies.")
        return _PyrokiAdapter(urdf_path, end_effector_link, **kwargs)
    elif adapter_type.lower() == 'curobo':
        if not _check_curobo():
            raise ImportError("curobo adapter not available. Please install curobo dependencies.")
        return _CuroboAdapter(urdf_path, end_effector_link, **kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")