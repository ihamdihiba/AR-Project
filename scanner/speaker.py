"""Step 6: offline text-to-speech (pyttsx3) on a dedicated thread.

pyttsx3 is not thread-safe, so one worker thread owns the engine and receives commands.
"""
import queue
import threading

import pyttsx3

from . import config

KEYWORDS = {
    "en": ("en_", "en-", "english"),
    "fr": ("fr_", "fr-", "french", "fran"),
    "ar": ("ar_", "ar-", "arabic", "arab", "hoda", "naayf", "maged", "tarik", "laila"),
}


class Speaker:
    def __init__(self):
        self._q = queue.Queue()
        self._voices = {}
        self._speaking = False
        self._ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self._ready.wait(8)

    @property
    def speaking(self) -> bool:
        return self._speaking

    def has_voice(self, lang: str) -> bool:
        return lang in self._voices

    def speak(self, text: str, lang: str) -> bool:
        """Queue speech. Returns False if no voice for that language is installed."""
        if lang not in self._voices:
            return False
        self._q.put(("say", text, lang))
        return True

    def stop(self):
        self._q.put(("stop",))

    def close(self):
        self._q.put(("quit",))

    @staticmethod
    def _match(voices, lang):
        for v in voices:
            blob = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
            if any(k in blob for k in KEYWORDS[lang]):
                return v.id
        return None

    def _loop(self):
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for lang in KEYWORDS:
                vid = self._match(voices, lang)
                if vid:
                    self._voices[lang] = vid
            if "en" not in self._voices:
                self._voices["en"] = engine.getProperty("voice")
        except Exception as e:  # noqa: BLE001
            print("Speech engine unavailable:", e)
            self._ready.set()
            return
        self._ready.set()

        engine.connect("finished-utterance", lambda name, completed: setattr(self, "_speaking", False))
        engine.startLoop(False)
        try:
            while True:
                try:
                    cmd = self._q.get(timeout=0.03)
                except queue.Empty:
                    cmd = None
                if cmd:
                    if cmd[0] == "quit":
                        break
                    if cmd[0] == "stop":
                        engine.stop()
                        self._speaking = False
                    elif cmd[0] == "say":
                        engine.stop()
                        engine.setProperty("voice", self._voices[cmd[2]])
                        engine.setProperty("rate", config.SPEECH_RATE)
                        engine.say(cmd[1])
                        self._speaking = True
                engine.iterate()
        finally:
            try:
                engine.endLoop()
            except Exception:  # noqa: BLE001
                pass
