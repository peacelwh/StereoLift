"""Registers ``policy.type=pi05_stereolift`` with the lerobot policy factory without editing lerobot.

Importing :mod:`stereolift` (or installing the ``.pth`` hook with ``scripts/install_plugin.sh``) before
``lerobot-train`` / ``lerobot-eval`` parse their configs is enough; ``python -m stereolift.train`` and
``python -m stereolift.eval`` do this for you.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

POLICY_TYPE = "pi05_stereolift"

try:
    import importlib
    import sys

    # The hook may run while ``lerobot.policies`` is still initialising; fetch the module object from sys.modules
    # first to avoid attribute lookups on a partially initialised package.
    _factory = sys.modules.get("lerobot.policies.factory") or importlib.import_module("lerobot.policies.factory")

    # importing the config registers it with draccus' ChoiceRegistry (PreTrainedConfig)
    from .configuration_pi05_stereolift import PI05StereoLiftConfig

    if not getattr(_factory, "_stereolift_patched", False):
        _orig_get_policy_class = _factory.get_policy_class
        _orig_make_policy_config = _factory.make_policy_config

        def get_policy_class(name: str):
            if name == POLICY_TYPE:
                from .modeling_pi05_stereolift import PI05StereoLiftPolicy

                return PI05StereoLiftPolicy
            return _orig_get_policy_class(name)

        def make_policy_config(policy_type: str, **kwargs):
            if policy_type == POLICY_TYPE:
                return PI05StereoLiftConfig(**kwargs)
            return _orig_make_policy_config(policy_type, **kwargs)

        _factory.get_policy_class = get_policy_class
        _factory.make_policy_config = make_policy_config
        _factory._stereolift_patched = True

        try:  # some call sites import the names directly from the package
            import lerobot.policies as _pkg

            if getattr(_pkg, "get_policy_class", None) is _orig_get_policy_class:
                _pkg.get_policy_class = get_policy_class
            if getattr(_pkg, "make_policy_config", None) is _orig_make_policy_config:
                _pkg.make_policy_config = make_policy_config
        except Exception:  # pragma: no cover
            pass
        log.info("StereoLift: registered policy type %r", POLICY_TYPE)
except ImportError as exc:  # lerobot not available in this interpreter
    log.debug("StereoLift plugin not registered: %s", exc)
except Exception as exc:  # pragma: no cover - never break the host process
    log.warning("StereoLift plugin registration failed: %r", exc)
