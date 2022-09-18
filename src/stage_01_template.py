import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import writetemplate
import random
import torch
from IPython.display import Image, clear_output  # to display images
import yaml

STAGE = "STAGE_NAME"

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def main():
    # clone YOLOv5 repository
    !git clone https: // github.com/ultralytics/yolov5.git  # clone repo
    %cd yolov5
    # !git reset --hard 886f1c03d839575afecb059accf74296fad395b6

    !pip install - qr requirements.txt  # install dependencies (ignore errors)

    # from utils.google_utils import gdrive_download  # to download models/datasets

    # clear_output()
    print('Setup complete. Using torch %s %s' % (torch.__version__,
          torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
    # this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
    print(% cat data.yaml)

    with open("data.yaml", 'r') as stream:
        num_classes = str(yaml.safe_load(stream)['nc'])

    print(num_classes)

    % % writetemplate / content/yolov5/models/custom_yolov5s.yaml

    nc: {num_classes}  # number of classes


depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
    - [10, 13, 16, 30, 33, 23]  # P3/8
    - [30, 61, 62, 45, 59, 119]  # P4/16
    - [116, 90, 156, 198, 373, 326]  # P5/32

# YOLOv5 backbone
backbone:
    # [from, number, module, args]
    [[-1, 1, Focus, [64, 3]],  # 0-P1/2
     [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
     [-1, 3, BottleneckCSP, [128]],
     [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
     [-1, 9, BottleneckCSP, [256]],
     [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
     [-1, 9, BottleneckCSP, [512]],
     [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
     [-1, 1, SPP, [1024, [5, 9, 13]]],
     [-1, 3, BottleneckCSP, [1024, False]],  # 9
     ]

# YOLOv5 head
head:
    [[-1, 1, Conv, [512, 1, 1]],
     [-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 6], 1, Concat, [1]],  # cat backbone P4
     [-1, 3, BottleneckCSP, [512, False]],  # 13

     [-1, 1, Conv, [256, 1, 1]],
     [-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 4], 1, Concat, [1]],  # cat backbone P3
     [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)

     [-1, 1, Conv, [256, 3, 2]],
     [[-1, 14], 1, Concat, [1]],  # cat head P4
     [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)

     [-1, 1, Conv, [512, 3, 2]],
     [[-1, 10], 1, Concat, [1]],  # cat head P5
     [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)

     [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
     ]

    % % time
%cd / content/yolov5/
!python train.py - -img 416 - -batch 16 - -epochs 300 - -data '../data.yaml' - -cfg ./models/custom_yolov5s.yaml - -weights 'yolov5s.pt' - -name yolov5s_results - -cache


if __name__ == "__main__":

    try:
        logging.info("\n***************************")
        logging.info(f">>>>>>>>>>> stage   {STAGE}   started <<<<<<<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>>>>>> stage  {STAGE}   completed <<<<<<<<<<<")
    except Exception as e:
        logging.exception(e)
        raise e
