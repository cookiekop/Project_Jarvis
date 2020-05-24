from scene import Object, Scene
from yolov3.models import Darknet  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import letterbox
from yolov3.utils.utils import *


class ObjectDetector:
    def __init__(self):
        self.img_size = 512
        self.augment = False
        self.half = False
        self.agnostic_nms = False
        self.iou_thres = 0.6
        self.fourcc = 'mp4v'
        self.conf_thres = 0.3
        self.out = 'output'
        self.save_txt = True
        self.view_img = True
        self.save_img = True

        weights = 'yolov3/weights/yolov3.pt'
        self.device = 'cuda'
        self.model = Darknet('yolov3/cfg/yolov3.cfg', self.img_size)
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()

        # Eval mode
        self.model.to(self.device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        # Get names and colors
        self.names = load_classes('yolov3/data/coco.names')

    def detect(self, img):
        # Run inference
        im0 = img.copy()

        # Padded resize
        img = letterbox(im0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
    
        # Inference
        t1 = torch_utils.time_synchronized()
        with torch.no_grad():
            pred = self.model(img, augment=self.augment)[0]
        t2 = torch_utils.time_synchronized()
        # print('Predict time: (%.3fs)' % (t2 - t1))
    
        # to float
        if self.half:
            pred = pred.float()
    
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   multi_label=False, classes=None, agnostic=self.agnostic_nms)
    
        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0)

        # Process detections
        det = pred[0]
        sce = Scene(im0)
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                obj = Object(self.names[int(cls)], xyxy, conf)
                sce.objs.append(obj)

        return sce
