# import torch
from tool.darknet2pytorch import Darknet
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from tool.torch_utils import do_detect

from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from dataset_coco import COCOImage

import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-anno_json', '--anno_json', type=str, default='./../MSCOCO_ObjDet_Detail/new_coco/annotations/new_anno.json', help='/path/coco/annotations/*.json')
parser.add_argument('-pred_json', '--pred_json', type=str, default='./../YOLOv4_pred.josn', help='/path/pred/annotations/*.json')
parser.add_argument('-img_path', '--img_path', type=str, default='./../MSCOCO_ObjDet_Detail/new_coco/images/', help='/path/coco/images/')
parser.add_argument('-cfgfile', '--cfgfile', type=str, default='./../cfg/yolov4.cfg', help='/path/*.cfg')
parser.add_argument('-weights', '--weights', type=str, default='./../weights/yolov4.weights', help='/path/.weights')
parser.add_argument('-use_cuda', '--use_cuda', type=int, default=1, help='0:use cpu, 1:use cuda')
parser.add_argument('-batch_size', '--batch_size', type=int, default=4, help='batch size')
opt = parser.parse_args()

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x

coco_category = coco80_to_coco91_class()

def coco_format(result):
    coco_pred = []
    img_id, boxes, H, W = result
    for box in boxes:
        x1, y1, x2, y2, conf, conf, category_id = box
        
        x1 = x1 * W
        y1 = y1 * H
        
        x2 = x2 * W
        y2 = y2 * H
        
        width = int(x2 - x1)
        height = int(y2 - y1)
        x, y = int(x1), int(y1)
        
        pred_out = {
            "image_id": int(img_id),
            "category_id": int(coco_category[category_id]),
#             "bbox": [x,y,width,height, int(x1), int(x2), int(y1), int(y2)],
            "bbox": [x,y,width,height],
            "score": float(conf),            
        }
        
        coco_pred.append(pred_out)
    return coco_pred

if __name__=='__main__':
    # Load Model
    m = Darknet(opt.cfgfile)
    m.load_weights(opt.weights)
    use_cuda = opt.use_cuda
    if use_cuda:
        m.cuda()

    # Data Loader
    anno = COCO(opt.anno_json)
    val_set = COCOImage(opt.anno_json, opt.img_path, 608)
    val_loader = DataLoader(val_set, opt.batchi_size, shuffle=True, num_workers=0)

    # Accumulate results
    result_dict = dict([])
    print('STRAT detection')
    for imgs, img_ids, sizes in tqdm(val_loader):
        # model
        boxes = do_detect(m, imgs, conf_thresh=0.4, nms_thresh=0.6, use_cuda=use_cuda, verbose=False)
        # process
        
        for img_id, box, H, W in zip(img_ids.numpy(), boxes, sizes[0].numpy(), sizes[1].numpy()):
            result_dict[img_id] = (img_id, box, H, W)

    # Transform results to COCO format
    print('Convert results to COCO format')
    total = []
    for img_id in tqdm(result_dict.keys()):
        one_result = coco_format(result_dict[img_id])
        total.extend(one_result)

    with open(opt.pred_json, 'w')as f:
        json.dump(total, f)

    # COCO Evaluation
    from pycocotools.cocoeval import COCOeval
    pred = anno.loadRes(opt.pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()