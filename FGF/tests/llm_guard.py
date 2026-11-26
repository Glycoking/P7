"""Patch helpers for the ``coding_fgf.llm`` package.

``coding_fgf.llm`` used to be one module, so patching ``coding_fgf.llm.call_structured_json``
intercepted every caller. It is now a package, and each submodule that calls a name reads its own
module-level binding. Patching only the package (or only one submodule) would silently miss the
other callers, and for "must not reach a live provider" guards that means a real API call.

These helpers patch *every* ``coding_fgf.llm`` submodule that binds the name, and fail loudly if
none does, so a rename cannot turn a guard into a no-op.
"""
from __future__ import annotations

import importlib
import pkgutil
from contextlib import ExitStack
from unittest.mock import patch

import coding_fgf.llm as llm_package


def llm_modules_binding(name: str) -> list:
    """Submodules of coding_fgf.llm that have a module-level attribute ``name``."""
    modules = []
    for info in pkgutil.iter_modules(llm_package.__path__):
        module = importlib.import_module(f"{llm_package.__name__}.{info.name}")
        if hasattr(module, name):
            modules.append(module)
    return modules


def patch_llm(monkeypatch, name: str, value) -> None:
    """monkeypatch.setattr on every llm submodule that binds ``name``."""
    modules = llm_modules_binding(name)
    assert modules, f"no coding_fgf.llm submodule binds {name!r}"
    for module in modules:
        monkeypatch.setattr(module, name, value)


def patch_llm_everywhere(name: str, value) -> ExitStack:
    """Context manager form for scripts: ``with patch_llm_everywhere(...):``."""
    modules = llm_modules_binding(name)
    assert modules, f"no coding_fgf.llm submodule binds {name!r}"
    stack = ExitStack()
    for module in modules:
        stack.enter_context(patch.object(module, name, value))
    return stack
