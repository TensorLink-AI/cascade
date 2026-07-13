"""Load a local ``.env`` into the process environment at CLI startup.

cascade reads every credential from the environment — never from the committed
``chain.toml`` (see :mod:`cascade.shared.hippius`): the Hippius Hub token, the
``HIPPIUS_S3_*`` keys, ``HF_TOKEN``, and now the ``BACKUP_S3_*`` R2-backup keys.
Operators commonly keep these in a gitignored ``.env`` (``.env`` and ``*.env``
are in ``.gitignore``); this helper loads that file so a plain ``.env`` "just
works" without a manual ``source``/``export`` step before launching a role.

Called once from each ``main()`` entry point. It never overrides a variable that
is already set, so an explicitly-exported value (CI, ``forward_env`` on a pod,
``docker --env-file``) always wins over the file. It is a no-op when
``python-dotenv`` is not installed or no ``.env`` is found, so nothing here is a
hard dependency of a running role.
"""

from __future__ import annotations

import os


def load_env_files() -> None:
    """Load ``.env`` (searching from CWD upward) into ``os.environ``.

    Existing environment variables are preserved (``override=False``). Silent and
    best-effort: a missing ``python-dotenv`` or a missing ``.env`` is a no-op, so
    a call at the top of a ``main()`` is always safe. ``CASCADE_NO_DOTENV=1``
    disables it entirely for callers that manage the environment themselves.
    """
    if os.environ.get("CASCADE_NO_DOTENV"):
        return
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
