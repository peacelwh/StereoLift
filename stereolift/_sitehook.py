"""Import hook that registers StereoLift the moment ``lerobot.policies.factory`` is imported.

Installed through a ``.pth`` line (``import stereolift._sitehook``) so that plain
``lerobot-train`` / ``lerobot-eval`` / real-robot ``run_policy.py`` see
``policy.type=pi05_stereolift`` without importing lerobot (or torch) at interpreter
start-up.  The hook is a no-op for processes that never import lerobot.
"""

from __future__ import annotations

import importlib.abc
import sys

_TARGET = "lerobot.policies.factory"


class _StereoLiftFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                import stereolift.lerobot_plugin  # noqa: F401
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"[stereolift] plugin registration failed: {exc!r}\n")

        loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _StereoLiftFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StereoLiftFinder())
