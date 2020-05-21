import cv2
from PIL import Image

from objectDetector import ObjectDetector
from scene import Text
from textDetector import TextDetector
from textReader import TextReader


class CVbox:
    def __init__(self):
        self._od = ObjectDetector()
        self._td = TextDetector()
        self._tr = TextReader()
        self._scene = None
    
    def run(self, img):
        self._scene = self._od.detect(img)
        text_boxes = self._td.detect(img)
        for i, box in enumerate(text_boxes):
            text_img = img[box[1]:box[3], box[0]:box[2]]
            str, conf = self._tr.read([(Image.fromarray(text_img).convert('L'), 'image')])
            text = Text(str, box, conf)
            self._scene.words.append(text)

cvbox = CVbox()
img = cv2.imread('./samples/sample.jpg')
cvbox.run(img)
cvbox._scene.order()
cvbox._scene.show()