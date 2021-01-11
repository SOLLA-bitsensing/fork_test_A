import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

import sys
import io
from contextlib import redirect_stdout

class capture(redirect_stdout):

    def __init__(self):
        self.f = io.StringIO()
        self._new_target = self.f
        self._old_targets = []  # verbatim from parent class

    def __enter__(self):
        self._old_targets.append(getattr(sys, self._stream))  # verbatim from parent class
        setattr(sys, self._stream, self._new_target)  # verbatim from parent class
        return self  # instead of self._new_target in the parent class

    def __repr__(self):
        return self.f.getvalue()  


parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--anno_json', type=str, default='./data/coco2017/annotations/instances_val2017.json', help='*.json path')
parser.add_argument('--pred_json', type=str, help='*.json path')

opt = parser.parse_args()
anno_json = opt.anno_json
pred_json = opt.pred_json

anno = COCO(anno_json)  # init annotations api
pred = anno.loadRes(pred_json)  # init predictions api
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
with capture() as msg:
    eval.summarize()
print(msg)

txt_name = pred_json[:-5] + '.txt'
with open(txt_name, 'w') as txt_file:
    txt_file.write(str(msg))