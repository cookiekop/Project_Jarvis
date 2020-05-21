import argparse
import string

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from OCR.dataset import AlignCollate
from OCR.model import Model
from OCR.utils import AttnLabelConverter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TextReader:
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--saved_model', default='OCR/weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'
                            , help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                            help='character label')
        parser.add_argument('--sensitive', default=True, action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                            help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1,
                            help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        self.opt = parser.parse_args()

        """ vocab / character number configuration """
        if self.opt.sensitive:
            self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)

        self.model = torch.nn.DataParallel(self.model).to(device)
        self.model.load_state_dict(torch.load(self.opt.saved_model, map_location=device))
        self.model.eval()
        self.collate_fn = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)

    def read(self, image):
        image_tensor, _ = self.collate_fn(image)
        with torch.no_grad():
            length_for_pred = torch.IntTensor([self.opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, self.opt.batch_max_length + 1).fill_(0).to(device)

            x = image_tensor.cuda()
            pred = self.model(x, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, pred_index = pred.max(2)
            pred_str = self.converter.decode(pred_index, length_for_pred)[0]
            pred_prob = F.softmax(pred, dim=2)
            pred_max_prob, _ = pred_prob.max(dim=2)
            pred_EOS = pred_str.find('[s]')
            pred_str = pred_str[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[0][:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        return pred_str, float(confidence_score)