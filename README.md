[HTCLite](https://www.lvisdataset.org/assets/challenge_reports/2020/MMDet.pdf) implementation using PyTorch (supports `MOSAIC` and `MixUp`)

#### Train

* Install [mmdet](https://github.com/open-mmlab/mmdetection) toolbox
* Register your dataset to `utils/dataset.py`, see `DAMAGEDataset`
* See `nets/exp01.py` for using `MOSAIC` and `MixUp` data pipeline
* See `nets/exp02.py` for using `RandomAugment` data pipeline
* Run `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs
#### Note
* The default configuration is for [2021 VIPriors Instance Segmentation Challenge](https://competitions.codalab.org/competitions/33340) dataset
#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection
