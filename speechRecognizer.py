from deepspeech import Model

class SpeechRecognizer:
    def __init__(self):
        self._model = Model('DeepSpeech/deepspeech-0.7.1-models.pbmm')
        # self._model.setBeamWidth(1)
        self._model.enableExternalScorer('DeepSpeech/deepspeech-0.7.1-models.scorer')

    def listen(self, audio):
        return self._model.stt(audio)