# yolov4
python test.py --cfg ./cfg/yolov4/yolov4.cfg --weights ./../weights/yolov4.weights --data data/coco2017.data --img-size 416 --device 1
python test.py --cfg ./cfg/yolov4/yolov4.cfg --weights ./../weights/yolov4.weights --data data/coco2017.data --img-size 512 --device 1
python test.py --cfg ./cfg/yolov4/yolov4.cfg --weights ./../weights/yolov4.weights --data data/coco2017.data --img-size 608 --device 1

# yolov3
python test.py --cfg ./cfg/yolov3/yolov3.cfg --weights ./../weights/yolov3.weights --data data/coco2017.data --img-size 416 --device 1
python test.py --cfg ./cfg/yolov3/yolov3.cfg --weights ./../weights/yolov3.weights --data data/coco2017.data --img-size 512 --device 1
python test.py --cfg ./cfg/yolov3/yolov3.cfg --weights ./../weights/yolov3.weights --data data/coco2017.data --img-size 608 --device 1

python coco_eval.py --pred_json results/results_yolov4_416.json
python coco_eval.py --pred_json results/results_yolov4_512.json
python coco_eval.py --pred_json results/results_yolov4_608.json

python coco_eval.py --pred_json results/results_yolov3_416.json
python coco_eval.py --pred_json results/results_yolov3_512.json
python coco_eval.py --pred_json results/results_yolov3_608.json
