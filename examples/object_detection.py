# pip install transformers
# pip install timm
# install pytorch based on you CUDA version
from PIL import Image
import numpy as np
import io
import aicard_eval
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from datasets import load_dataset
import pprint

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

dataset = load_dataset("rishitdagli/cppe-5", split='test')

processor = DetrImageProcessor.from_pretrained('devonho/detr-resnet-50_finetuned_cppe5')
model = DetrForObjectDetection.from_pretrained('devonho/detr-resnet-50_finetuned_cppe5').to('cuda')

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

def pipeline(data):
    images = [Image.open(io.BytesIO(img['bytes'])).convert("RGB") for img in data['image']]

    inputs = processor(images=images, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1] for image in images]).to('cuda')
    resultss = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)
    
    out = []
    for results in resultss:
        out.append({"boxes": [xyxy_to_xywh(box) for box in results["boxes"].cpu().tolist()],
                    "labels": results["labels"].cpu().tolist(),
                    "scores": results["scores"].cpu().tolist()}) 
    return out
    

metrics = aicard_eval.evaluate(
    data=dataset,
    pipeline=pipeline,
    task=aicard_eval.tasks.vision.object_detection,
    batch_size=5,
    box_format='xywh')

pprint.pprint(metrics)
# {'batch_size': 5,
#  'datetime': '2025-Nov-20 15:56',
#  'execution_time': 'inference: 2.39s, metrics: 29.88ms',
#  'hardware': 'CPU: AMD Ryzen 7 7800X3D 8-Core Processor, RAM: 15.62 GB, CUDA: '
#              '| NVIDIA-SMI 580.102.01             Driver Version: '
#              '581.57         CUDA Version: 13.0     |',
#  'metrics': {'od_metrics': {'class_metrics': {0: {'AP@[.5 | all | 100]': 0.6547029702970297,
#                                                   'AP@[.5:.95 | all | 100]': 0.4642151886442633,
#                                                   'AP@[.5:.95 | large | 100]': 0.4741592650076004,
#                                                   'AP@[.5:.95 | medium | 100]': 0.0,
#                                                   'AP@[.5:.95 | small | 100]': -1.0,
#                                                   'AP@[.75 | all | 100]': 0.5053356792566358,
#                                                   'AR@[.5 | all | 100]': 0.6666666666666666,
#                                                   'AR@[.5:.95 | all | 100]': 0.5155555555555555,
#                                                   'AR@[.5:.95 | all | 10]': 0.5155555555555555,
#                                                   'AR@[.5:.95 | all | 1]': 0.3888888888888889,
#                                                   'AR@[.5:.95 | large | 100]': 0.5272727272727272,
#                                                   'AR@[.5:.95 | medium | 100]': 0.0,
#                                                   'AR@[.5:.95 | small | 100]': -1.0,
#                                                   'AR@[.75 | all | 100]': 0.5555555555555556},
#                                               1: {'AP@[.5 | all | 100]': 0.4084158415841584,
#                                                   'AP@[.5:.95 | all | 100]': 0.21683168316831683,
#                                                   'AP@[.5:.95 | large | 100]': 0.24759075907590758,
#                                                   'AP@[.5:.95 | medium | 100]': 0.17673267326732672,
#                                                   'AP@[.5:.95 | small | 100]': -1.0,
#                                                   'AP@[.75 | all | 100]': 0.1188118811881188,
#                                                   'AR@[.5 | all | 100]': 0.4117647058823529,
#                                                   'AR@[.5:.95 | all | 100]': 0.2352941176470588,
#                                                   'AR@[.5:.95 | all | 10]': 0.2352941176470588,
#                                                   'AR@[.5:.95 | all | 1]': 0.2352941176470588,
#                                                   'AR@[.5:.95 | large | 100]': 0.2636363636363636,
#                                                   'AR@[.5:.95 | medium | 100]': 0.18333333333333332,
#                                                   'AR@[.5:.95 | small | 100]': -1.0,
#                                                   'AR@[.75 | all | 100]': 0.23529411764705882},
#                                               2: {'AP@[.5 | all | 100]': 0.20367751060820366,
#                                                   'AP@[.5:.95 | all | 100]': 0.07306164545025931,
#                                                   'AP@[.5:.95 | large | 100]': 0.0,
#                                                   'AP@[.5:.95 | medium | 100]': 0.10119711971197118,
#                                                   'AP@[.5:.95 | small | 100]': 0.0,
#                                                   'AP@[.75 | all | 100]': 0.022277227722772276,
#                                                   'AR@[.5 | all | 100]': 0.26229508196721313,
#                                                   'AR@[.5:.95 | all | 100]': 0.1180327868852459,
#                                                   'AR@[.5:.95 | all | 10]': 0.1180327868852459,
#                                                   'AR@[.5:.95 | all | 1]': 0.060655737704918035,
#                                                   'AR@[.5:.95 | large | 100]': 0.0,
#                                                   'AR@[.5:.95 | medium | 100]': 0.1565217391304348,
#                                                   'AR@[.5:.95 | small | 100]': 0.0,
#                                                   'AR@[.75 | all | 100]': 0.08196721311475409},
#                                               3: {'AP@[.5 | all | 100]': 0.0,
#                                                   'AP@[.5:.95 | all | 100]': 0.0,
#                                                   'AP@[.5:.95 | large | 100]': 0.0,
#                                                   'AP@[.5:.95 | medium | 100]': 0.0,
#                                                   'AP@[.5:.95 | small | 100]': 0.0,
#                                                   'AP@[.75 | all | 100]': 0.0,
#                                                   'AR@[.5 | all | 100]': 0.0,
#                                                   'AR@[.5:.95 | all | 100]': 0.0,
#                                                   'AR@[.5:.95 | all | 10]': 0.0,
#                                                   'AR@[.5:.95 | all | 1]': 0.0,
#                                                   'AR@[.5:.95 | large | 100]': 0.0,
#                                                   'AR@[.5:.95 | medium | 100]': 0.0,
#                                                   'AR@[.5:.95 | small | 100]': 0.0,
#                                                   'AR@[.75 | all | 100]': 0.0},
#                                               4: {'AP@[.5 | all | 100]': 0.36724949967524223,
#                                                   'AP@[.5:.95 | all | 100]': 0.16702162880641253,
#                                                   'AP@[.5:.95 | large | 100]': 0.1933993399339934,
#                                                   'AP@[.5:.95 | medium | 100]': 0.21071831763008234,
#                                                   'AP@[.5:.95 | small | 100]': 0.09438943894389439,
#                                                   'AP@[.75 | all | 100]': 0.12433770849612433,
#                                                   'AR@[.5 | all | 100]': 0.4230769230769231,
#                                                   'AR@[.5:.95 | all | 100]': 0.21346153846153845,
#                                                   'AR@[.5:.95 | all | 10]': 0.21346153846153845,
#                                                   'AR@[.5:.95 | all | 1]': 0.16346153846153846,
#                                                   'AR@[.5:.95 | large | 100]': 0.22142857142857145,
#                                                   'AR@[.5:.95 | medium | 100]': 0.268,
#                                                   'AR@[.5:.95 | small | 100]': 0.1,
#                                                   'AR@[.75 | all | 100]': 0.21153846153846154}},
#                             'classes': [0, 1, 2, 3, 4],
#                             'mAP@[.5 | all | 100]': 0.32680916443292685,
#                             'mAP@[.5:.95 | all | 100]': 0.18422602921385037,
#                             'mAP@[.5:.95 | large | 100]': 0.18302987280350028,
#                             'mAP@[.5:.95 | medium | 100]': 0.09772962212187607,
#                             'mAP@[.5:.95 | small | 100]': 0.031463146314631464,
#                             'mAP@[.75 | all | 100]': 0.1541524993327303,
#                             'mAR@[.5 | all | 100]': 0.35276067551863116,
#                             'mAR@[.5:.95 | all | 100]': 0.21646879970987976,
#                             'mAR@[.5:.95 | all | 10]': 0.21646879970987976,
#                             'mAR@[.5:.95 | all | 1]': 0.16966005654048086,
#                             'mAR@[.5:.95 | large | 100]': 0.20246753246753246,
#                             'mAR@[.5:.95 | medium | 100]': 0.12157101449275362,
#                             'mAR@[.5:.95 | small | 100]': 0.03333333333333333,
#                             'mAR@[.75 | all | 100]': 0.21687106957116603,
#                             'n_images': 29}},
#  'package version': '0.1.0',
#  'task': 'Object Detection'}