"""Utilities."""

import functools
import logging
import os
import subprocess
import urllib.request
from datetime import date
from pathlib import Path

_DIR = Path(__file__).parent
PDDL_DIR = _DIR / "envs" / "assets" / "pddl"


@functools.lru_cache(maxsize=None)
def get_git_commit_hash() -> str:
    """Return the hash of the current git commit."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode("ascii").strip()


def get_pddl_from_url(url: str, cache_dir: Path = PDDL_DIR) -> str:
    """Download a PDDL file from a given URL.

    If the file already exists in the cache_dir, load instead.

    Note that this assumes the PDDL won't change at the URL.
    """
    sanitized_url = "".join(x for x in url if x.isalnum())
    file_name = f"cached-pddl-{sanitized_url}"
    file_path = cache_dir / file_name
    # Download if doesn't already exist.
    if not os.path.exists(file_path):
        logging.info(f"Cache not found for {url}, downloading.")
        with urllib.request.urlopen(url) as f:
            pddl = f.read().decode('utf-8')
        if "(:action" not in pddl and "(:init" not in pddl:
            raise ValueError(f"PDDL file not found at {url}.")
        # Add a note at the top of the file about when this was downloaded.
        today = date.today().strftime("%B %d, %Y")
        note = f"; Downloaded {today} from {url}\n"
        pddl = note + pddl
        # Cache.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pddl)
    with open(file_path, "r", encoding="utf-8") as f:
        pddl = f.read()
    return pddl
