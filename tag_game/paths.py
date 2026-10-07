"""Paths written into result files (and echoed to stdout).

A results JSON must not carry a machine-specific absolute path (it would leak
the local directory layout into a committed file). Every writer that records a
model, config or run path passes it through `repo_relative`:

  * an absolute path inside this repository becomes repo-root-relative
    (POSIX separators);
  * an absolute path outside the repository under the home directory has the
    home prefix replaced with `~`;
  * anything else — a relative path, a non-path keyword — is written exactly
    as given, so existing outputs are unchanged.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _home_tilde(s):
    home = os.path.expanduser("~").rstrip(os.sep)
    if home and home != "~" and (s == home or s.startswith(home + os.sep)):
        return "~" + s[len(home):]
    return s


def repo_relative(path):
    if path is None:
        return None
    s = str(path)
    if not os.path.isabs(s):
        return s
    try:
        return Path(s).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return _home_tilde(s)


def repo_relative_spec(spec):
    """Policy/opponent specs are a keyword ('heuristic', 'random', ...), a
    path, or 'model:<path>'; only the path part is rewritten."""
    if not isinstance(spec, str):
        return spec
    if spec.startswith("model:"):
        return "model:" + repo_relative(spec[len("model:"):])
    return repo_relative(spec)
