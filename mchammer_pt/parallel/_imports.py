"""Validation helpers for spawn-pool class and callable arguments.

Spawn-mode workers re-import any class or callable argument by fully
qualified name. Classes and callables defined inside a function or in
an interactive ``__main__`` cannot be re-imported. The helper here
rejects those up-front with a clear ``ValueError`` rather than letting
the failure surface as a deep multiprocessing ``PicklingError`` or an
``EOFError`` from the parent's first ``recv()``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal


def _resolve_replicas(
    replicas: Sequence[int] | Literal["all"],
    n_replicas: int,
) -> list[int]:
    """Convert and range-check ``replicas`` into a list of indices.

    Args:
        replicas: either the string ``"all"`` or a sequence of integer
            indices into the pool's replica list.
        n_replicas: total number of replicas in the pool.

    Returns:
        The selected indices as a list, in the order the caller
        supplied, with duplicates removed (passing ``[0, 0]`` is
        equivalent to ``[0]``). An empty list is returned for an empty
        input sequence; callers short-circuit on that.

    Raises:
        IndexError: if any index falls outside ``range(n_replicas)``.
    """
    if replicas == "all":
        return list(range(n_replicas))
    out = [int(i) for i in replicas]
    for i in out:
        if not 0 <= i < n_replicas:
            raise IndexError(
                f"replica index {i} out of range for pool of size {n_replicas}"
            )
    return list(dict.fromkeys(out))


def _check_importable(obj: type | Callable[..., Any], *, kind: str) -> None:
    """Reject ``obj`` definitions spawn workers cannot re-import.

    ``obj`` can be a class, a function/method, or an instance of a
    user-defined callable class. For instances the check is applied to
    ``type(obj)`` because that is what pickle walks to find the class
    for reconstruction.

    Two definition sites break the FQN-import contract:

    1. **Interactive ``__main__``.** An object defined in a Jupyter cell
       or REPL has ``__module__ == "__main__"``, but
       ``sys.modules["__main__"]`` has no importable ``.py`` file.
       Top-level objects in ``python script.py`` are fine (the worker
       re-runs the script as ``__main__``); the discriminator is
       whether ``__main__.__file__`` ends in ``.py``.
    2. **Function-local object.** An object defined inside a function has
       ``"<locals>"`` in its ``__qualname__``. Pickle cannot walk into
       a function's local scope to recover the object.

    Args:
        obj: the class, callable, or callable instance to check.
        kind: short human-readable noun phrase used in error messages
            (e.g. ``"ensemble_cls"`` or ``"observer class"``).

    Raises:
        ValueError: if ``obj`` cannot be re-imported by a spawn worker.
    """
    # Resolve the importable target. Functions and classes carry
    # __qualname__ directly; instances of user-defined callable classes
    # do not, so fall through to the class.
    target = obj if hasattr(obj, "__qualname__") else type(obj)
    if "<locals>" in target.__qualname__:
        raise ValueError(
            f"{kind}={target.__qualname__!r} is defined inside a function, "
            f"so spawn workers cannot re-import it. Move it to module "
            f"top level (or make it a method of a top-level class)."
        )
    if target.__module__ != "__main__":
        return
    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is not None and main_file.endswith(".py"):
        return
    raise ValueError(
        f"{kind}={target.__name__!r} is defined in __main__ in a session "
        f"whose __main__ cannot be re-imported by spawn workers "
        f"(typically Jupyter or a REPL). Move it into a .py module that "
        f"both your session and the workers can import."
    )
