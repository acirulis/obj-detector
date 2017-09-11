import pyyolo
import numpy as np
import json

class Pyyolo:

    darknet_path = './darknet'
    datacfg = 'andis/andis.data'
    cfgfile = 'andis/andis.cfg'
    weightfile = 'andis/andis_40000.weights'


   # datacfg = 'cfg/coco2.data'
   # weightfile = 'yolo.weights'
   # cfgfile = 'cfg/yolo.cfg'

    thresh = 0.35
    hier_thresh = 1


    def __init__(self):
        pyyolo.init(self.darknet_path, self.datacfg, self.cfgfile, self.weightfile)


    # def __del__(self):
    #     pyyolo.cleanup()

    def test(self, filename):
        out = pyyolo.test(filename, self.thresh, self.hier_thresh, 0)
        return out

    def recon(self, img):
        img = img.transpose(2, 0, 1)
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        # print w, h, c
        data = img.ravel() / 255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        outputs = pyyolo.detect(w, h, c, data, self.thresh, self.hier_thresh)
        return outputs
