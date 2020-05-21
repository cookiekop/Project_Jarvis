from collections import OrderedDict

import cv2
import torch
from torch.autograd import Variable

import craft_utils
import imgproc
from craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def box2xyxy(box, shape):
    x_min = 99999
    x_max = 0
    y_min = 99999
    y_max = 0
    for point in box:
        if point[0] < x_min:
            x_min = int(point[0])
        if point[0] > x_max:
            x_max = int(point[0])
        if point[1] < y_min:
            y_min = int(point[1])
        if point[1] > y_max:
            y_max = int(point[1])
    x_min = max(0, x_min)
    x_max = min(shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(shape[0], y_max)
    return [x_min, y_min, x_max, y_max]

class TextDetector:
    def __init__(self):
        #Parameters
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.text_threshold = 0.7
        self.low_text = 0.4
        self.link_threshold = 0.4
        self.refine = False
        self.refiner_model = ''
        self.poly = False

        self.net = CRAFT()
        self.net.load_state_dict(copyStateDict(torch.load('CRAFT/weights/craft_mlt_25k.pth')))
        self.net = self.net.cuda()
        self.net = torch.nn.DataParallel(self.net)
        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if self.refine:
            from refinenet import RefineNet
            self.refine_net = RefineNet()
            self.refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model)))
            self.refine_net = self.refine_net.cuda()
            self.refine_net = torch.nn.DataParallel(self.refine_net)
            self.refine_net.eval()
            self.poly = True

    def detect(self, image):
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

        x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()


        # Post-processing
        boxes, _ = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold,
                                               self.low_text, self.poly)
        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        toRet = []
        for box in boxes:
            toRet.append(box2xyxy(box, image.shape[0: 2]))

        return toRet


