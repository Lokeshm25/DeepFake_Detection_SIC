"""Process and thread locking for append logs and provenance publication."""

import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

_FILE_LOCK = threading.RLock()


@contextmanager
def _locked_text_file(path: Path) -> Iterator[TextIO]:
    with _FILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8", newline="") as stream:
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield stream
            finally:
                stream.flush()
                stream.seek(0)
                if sys.platform == "win32":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
