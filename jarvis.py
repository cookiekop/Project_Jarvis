from audiobox import AudioBox
from cvbox import CVbox


class Jarvis:
    def __init__(self):
        self._vision = CVbox()
        self._hearing = AudioBox()