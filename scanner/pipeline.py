"""Runs the scan pipeline (OCR -> fuzzy match -> online fallback) in a background thread
so the camera preview and hand tracking never freeze."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from . import ocr_engine


@dataclass
class ScanState:
    status: str = "idle"          # idle | scanning | found | notfound | error
    medicine: dict | None = None
    source: str = ""              # local | openfda | ai
    ocr_text: str = ""
    detail: str = ""
    stamp: float = field(default_factory=time.time)


class Scanner:
    def __init__(self, db):
        self.db = db
        self.state = ScanState()
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, snapshot) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self.state = ScanState(status="scanning")
        threading.Thread(target=self._run, args=(snapshot,), daemon=True).start()
        return True

    def _run(self, snapshot):
        try:
            text = ocr_engine.read_text(snapshot)
            med, _score = self.db.find(text)
            source = "local"
            if med is None and text.strip():
                med, source = self._online(text)
            if med:
                self.state = ScanState("found", med, source, text)
            else:
                self.state = ScanState("notfound", None, "", text)
        except Exception as e:  # noqa: BLE001 - show any problem on screen instead of crashing
            self.state = ScanState("error", None, "", "", detail=f"{type(e).__name__}: {e}"[:140])
        finally:
            self._busy = False

    def _online(self, text: str):
        return None, ""
