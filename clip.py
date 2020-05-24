from configurations import *
from scipy.io.wavfile import write
import cv2
import json

class Clip:
    def __init__(self, audio):
        self.scenes = []
        self.audio = audio
        self.sounds = None
        self.narative = None

    def save(self):
        self_hash = str(hash(self))
        write('MemoryFiles/audio/'+self_hash+'.wav', AUDIO_FPS, self.audio)
        json_str = '{"scenes": ['
        for i, sce in enumerate(self.scenes):
            json_str += sce.save()
            if i < len(self.scenes) - 1:
                json_str += ', '
        json_str += '], \n'
        json_str += '"audio": ' + self_hash + ', \n'
        json_str += '"sounds": ['
        for i, sound in enumerate(self.sounds):
            json_str += '{"cls": "' + sound[0] + '", "conf": ' + str(sound[1]) + '}'
            if i < len(self.sounds) - 1:
                json_str += ', '
        json_str += '], \n'
        json_str += '"narative": "' + self.narative + '"}'
        with open('MemoryFiles/' + self_hash + '.json', 'w') as f:
            f.write(json_str)