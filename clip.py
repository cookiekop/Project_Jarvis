from configurations import *
from scipy.io.wavfile import write
import cv2
import numpy as np

class Clip:
    def __init__(self, scene, audio):
        self.scene = scene
        self.audio = audio
        self.sounds = None
        self.narative = None

    def save(self):
        self_hash = str(hash(self))
        write('MemoryFiles/audio/'+self_hash+'.wav', AUDIO_FPS, self.audio)
        json_str = '{"scene": ' + self.scene.save() + ', \n'
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


class Scene:
    def __init__(self, img):
        self.img = img
        self.objs = []
        self.words = []

        self.center = np.array(img.shape[:2]) / 2

    def show(self):
        img_toshow = self.img.copy()
        for obj in self.objs:
            img_toshow = cv2.rectangle(img_toshow, (int(obj.box[0]), int(obj.box[1])),
                                        (int(obj.box[2]), int(obj.box[3])), (255,0,0))
            # cv2.imshow('window', img_toshow)
            # cv2.waitKey(0)

        for i, word in enumerate(self.words):
            img_toshow = cv2.rectangle(img_toshow, (int(word.box[0]), int(word.box[1])),
                                       (int(word.box[2]), int(word.box[3])), (0, 255, 0))
            # cv2.imshow('window', img_toshow)
            # cv2.waitKey(0)

    def order(self):
        self.objs.sort(key=lambda k:k.conf * (k.size + np.sum(np.power(k.center - self.center, 2))), reverse=True)

        # reading_vec = np.array([self.img.shape[2], self.img.shape[1]])
        # self.words.sort(key=lambda k: np.sqrt(np.sum(np.power(k.center, 2) * reading_vec)))

    def save(self):
        sce_hash = str(hash(self))
        sce_json = '{"objs": ['
        for j, obj in enumerate(self.objs):
            xyxy = [int(element.item()) for element in obj.box]
            sce_json += '{"name": "' + obj.name + '", "box": ' + str(xyxy) + \
                        ', "conf": ' + str(float(obj.conf)) + '}'
            if j < len(self.objs) - 1:
                sce_json += ', '
        sce_json += '], "words": ['
        for j, word in enumerate(self.words):
            sce_json += '{"str": "' + word.str + '", "box": ' + str(word.box) + \
                        ', "conf": ' + str(float(word.conf)) + '}'
            if j < len(self.words) - 1:
                sce_json += ', '
        sce_json += '] }'
        cv2.imwrite('MemoryFiles/scene/image/' + sce_hash + '.png', self.img)
        with open('MemoryFiles/scene/' + sce_hash + '.json', 'w') as f:
            f.write(sce_json)
        return sce_hash

class Object:
    def __init__(self, name, box, conf):
        self.name = name
        self.box = box
        self.conf = conf

        self.size = abs(self.box[0] - self.box[2]) * abs(self.box[1] - self.box[3])
        center = ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)
        self.center = np.array(center)

class Text:
    def __init__(self, str, box, conf):
        self.str = str
        self.box = box
        self.conf = conf

        center = ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)
        self.center = np.array(center)