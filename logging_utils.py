"""Small logging helper so every module logs in one consistent format."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        level = os.environ.get("PYRAGUARD_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
