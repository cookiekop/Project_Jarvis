import cv2
import numpy as np

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