import json
import os
import pandas as pd
import cv2
# Check Pytorch installation
import torch, torchvision
from flask import Flask, Response, Request, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from loguru import logger
import base64
from io import BytesIO
from PIL import Image


from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import set_random_seed

# Imports
import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

import random
import numpy as np
from pathlib import Path
import mmcv

from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, init_detector, inference_detector
import torch 

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmcv import Config
from mmdet.apis import set_random_seed
cfg = Config.fromfile('/home/ubuntu/kaggle/sartorius/utils/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py')

cfg.backbone = dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False)


# modify num classes of the model in box head
cfg.model.roi_head.bbox_head[0].num_classes = 1
cfg.model.roi_head.bbox_head[1].num_classes = 1
cfg.model.roi_head.bbox_head[2].num_classes = 1

cfg.model.roi_head.mask_head.num_classes = 1

checkpoint_file = "/home/ubuntu/zhuldyzzhan/epoch_12.pth"

# build the model from a config file and a checkpoint file
model = init_detector(cfg, checkpoint_file, device='cuda:0')
model.CLASSES = ("Note", )

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def img_to_b64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

@app.route("/index", methods=["GET"])
def index():
    return Response(status=200)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']

    filename = secure_filename(file.filename)
    img_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(img_name)

    img = mmcv.imread(img_name)
    model.cfg = cfg
    result = inference_detector(model, img)
    rects = []

    for r in result[0][0]:
        r = [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
        rects.append(r)

        cv2.rectangle(img,(int(r[0]), int(r[1])),(int(r[2]), int(r[3])),(0,255,0),3)
    
    im = Image.fromarray(img)
    b64 = img_to_b64(im)
    img_base64 = bytes("data:image/jpeg;base64,", encoding='utf-8') + b64

    resp = {
        "rects": rects,
        "base64": img_base64.decode("ascii")
    }

    return jsonify(resp)

# img = mmcv.imread("21_guns-1_png.rf.3a41598ab5ea066c84c33efba3a46101.jpg")
# model.cfg = cfg
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.5) 

def main():
    app.run(host="0.0.0.0", port=8888)


if __name__ == '__main__':
    main()