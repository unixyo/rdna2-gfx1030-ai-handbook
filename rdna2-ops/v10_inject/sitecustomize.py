"""Research-only import hook; no torch import and no operating file edits."""
import importlib.abc
import importlib.machinery
import os
import sys

class _LabLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original
    def create_module(self, spec):
        method = getattr(self.original, "create_module", None)
        return method(spec) if method else None
    def exec_module(self, module):
        self.original.exec_module(module)
        from v10_core_hook import install
        install(module)

class _LabFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "rvc.train.lab_hooks":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _LabLoader(spec.loader)
        return spec

if os.environ.get("APPLIO_V10_OUTPUT"):
    sys.meta_path.insert(0, _LabFinder())
