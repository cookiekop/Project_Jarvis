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
        print(words)

audiobox = AudioBox()
wav_data, sr = sf.read('samples/appropriate_respect.wav', dtype=np.int16)

if len(wav_data.shape) > 1:
    wav_data = np.mean(wav_data, axis=1)
if sr != 16000:
    wav_data = resampy.resample(wav_data, sr, 16000)

audiobox.run(wav_data)