"""Kokoro text-to-speech (offline). The model is loaded once, on the first call."""
import threading

import soundfile as sf

from core.settings import ROOT

MODEL = ROOT / "assets" / "voices" / "kokoro-v1.0.onnx"
VOICES = ROOT / "assets" / "voices" / "voices-v1.0.bin"
VOICE_NAMES = ["af_heart", "af_bella", "am_michael", "bf_emma"]

_k = None
_lock = threading.Lock()


def files_ok() -> bool:
    return MODEL.exists() and VOICES.exists()


def synth(text, out_wav, voice="af_heart", speed=1.0) -> float:
    """Writes a wav and returns its length in seconds."""
    global _k
    if not files_ok():
        raise FileNotFoundError("Kokoro files are missing in assets/voices/ (kokoro-v1.0.onnx and voices-v1.0.bin)")
    with _lock:
        if _k is None:
            from kokoro_onnx import Kokoro
            _k = Kokoro(str(MODEL), str(VOICES))
        samples, rate = _k.create(text, voice=voice, speed=speed, lang="en-us")
    sf.write(str(out_wav), samples, rate)
    return len(samples) / rate