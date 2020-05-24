import numpy as np
import resampy
import soundfile as sf

from soundDetector import SoundDetector
from speechRecognizer import SpeechRecognizer


class AudioBox:
    def __init__(self):
        self._sd = SoundDetector()
        self._sr = SpeechRecognizer()
    def run(self, waveform):
        sounds = self._sd.listen(waveform)
        words = self._sr.listen(waveform)
        return sounds, words