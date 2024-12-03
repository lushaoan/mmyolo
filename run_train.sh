mim install -e /dataset/shaoanlu/github/mmlab/mmdetection/
pip uninstall mmcv
mim install mmcv==2.0.0rc4
pip install prettytable
pip install albumentations==1.3.1

pip list

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ./configs/barcode/yolov8/exp2.py 4 --work-dir ./work_dirs/exp2/