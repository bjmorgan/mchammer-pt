"""Validation helpers for spawn-pool class arguments.

Spawn-mode workers re-import any class argument (ensemble class,
observer class) by fully qualified name. Classes defined inside a
function or in an interactive ``__main__`` cannot be re-imported. The
helper here rejects those up-front with a clear ``ValueError`` rather
than letting the failure surface as a deep multiprocessing
``PicklingError`` or an ``EOFError`` from the parent's first
``recv()``.
"""

from __future__ import annotations

import sys


def _check_class_importable(cls: type, *, kind: str) -> None:
    """Reject ``cls`` definitions spawn workers cannot re-import.

    Two definition sites break the FQN-import contract:

    1. **Interactive ``__main__``.** A class defined in a Jupyter cell
       or REPL has ``__module__ == "__main__"``, but
       ``sys.modules["__main__"]`` has no importable ``.py`` file.
       Top-level classes in ``python script.py`` are fine (the worker
       re-runs the script as ``__main__``); the discriminator is
       whether ``__main__.__file__`` ends in ``.py``.
    2. **Function-local class.** A class defined inside a function has
       ``"<locals>"`` in its ``__qualname__``. Pickle cannot walk into
       a function's local scope to recover the class.

    Args:
        cls: the class to check.
        kind: short human-readable noun phrase used in error messages
            (e.g. ``"ensemble_cls"`` or ``"observer class"``).

    Raises:
        ValueError: if ``cls`` cannot be re-imported by a spawn worker.
    """
    if "<locals>" in cls.__qualname__:
        raise ValueError(
            f"{kind}={cls.__qualname__!r} is defined inside a function, "
            f"so spawn workers cannot re-import it. Move the class to "
            f"module top level (or to a method of a top-level class)."
        )
    if cls.__module__ != "__main__":
        return
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is not None and main_file.endswith(".py"):
        return
    raise ValueError(
        f"{kind}={cls.__name__!r} is defined in __main__ in a session "
        f"whose __main__ cannot be re-imported by spawn workers "
        f"(typically Jupyter or a REPL). Move the class into a .py "
        f"module that both your session and the workers can import."
    )
