## 큰그림 그려놓기 or 생각나는 것들 마구잡이로 적어놓는 란

- 해야되는 일
- 필요하다고 생각되는 기능
- 기록해둘 필요가 있는것들
- 주요 마일스톤 등등을 정리 및 기록

### 해야될일
- [x] Pytorch YOLOv4 darknet wegith, cfg parsing model
    - https://github.com/bitsensing/YOLO_Pytorch/blob/master/pytorch-YOLOv4/modeling_cfg.ipynb
- [x] Pytorch YOLOv4 pretrained model evaluation check
    - 평가는 mAP 기준, COCO API를 사용
    - https://github.com/bitsensing/YOLO_Pytorch/blob/master/MSCOCO_ObjDet_Detail/pycocoEvalDemo.ipynb
- [ ] Pytorch YOLOv4 model train
    - darknet weight 그대로 가져와서 train 시켜보고 성능 잘 나오는지 확인
    - fine tuned 할 부분만 정해서 돌려보기
- [ ] Pytorch YOLOv4 weight -> Darknet weight convertor
    - weight file format 체크 필요
    - pytorch weight parsing 하여 weight로 변환 하는 부분 필요
- [ ] Pytorch YOLOv4 model -> Darknet .cfg convertor?
    - 만약 모델을 건드리게되면 필요할까?? TensorRT engine을 deep stream sdk가 아닌 외부에서 build 한다면 필요할듯??
    - custom data training으로 class 개수를 변환하게 된다면 cfg를 조금 건드려야 될수도 있음
- [ ] Deep stream sdk에서 잘 돌아가는지 확인
    - custom data로 학습시킨 후 weight file 넘겨서 잘 돌아가는지 확인

- [ ] ATM data format 확인 및 COCO & YOLO format으로 변환
    - labeling format도 변환 필요 (e.g. YOLO label format)
- [ ] ATM data로 train 및 평가

### Model Format
- Pytorch YOLOv4는 Darknet 원저자 repo 모델 format을 따라감(deep stream sdk에서 돌아가야되서) (추후 modify 가능하면 modify)
    - cfg (e.g. yolov4.cfg)
    - weights (e.g. yolov4.weights)

### Data Format
- YOLO Format
    - YOLO format dataset create : https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data#2-create-labels

- COCO Format
    - coco format 설명 : https://github.com/bitsensing/YOLO_Pytorch/tree/master/MSCOCO_ObjDet_Detail
- Data input output은 Darknet format 그대로 사용
    - input image 는 (0, 1) 로 normalize (i.e. (rgb / 255).float())
    - output 정확한 format은 체크 필요
        - bbox foramt (x1, y1, x2, y2) <- 확인필요
        - nms 결과 data format(shape 및 datatype 포함) 등 체크 

### Evaluation
**Evaluation 기록시 성능에 영향을 미치는 Hyper Parameters는 전부 기록하는것을 지향**

**대표적으로 nms confidence score threshold & iou threshold**

- 평가는 mAP 기준, COCO API를 사용
    - https://github.com/bitsensing/YOLO_Pytorch/blob/master/MSCOCO_ObjDet_Detail/pycocoEvalDemo.ipynb




