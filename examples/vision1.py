import aicard_eval
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import io
import pprint

model_name = "prithivMLmods/Mnist-Digits-SigLIP2"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

dataset = load_dataset("ylecun/mnist", split='test')

def pipeline(data):
    image = data['image'][0]['bytes']
    image = Image.open(io.BytesIO(image)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    return [probs]


metrics = aicard_eval.evaluate(
    data=dataset.select(range(100)),
    pipeline=pipeline,
    task=aicard_eval.tasks.vision.image_classification)

pprint.pprint(metrics)
# {'batch_size': 1,
#  'datetime': '2025-Nov-21 12:26',
#  'execution_time': 'inference: 9.36s, metrics: 11.42ms',
#  'hardware': 'CPU: AMD Ryzen 7 7800X3D 8-Core Processor, RAM: 15.62 GB, CUDA: '
#              '| NVIDIA-SMI 580.102.01             Driver Version: '
#              '581.57         CUDA Version: 13.0     |',
#  'metrics': {'auc_roc_macro': 1.0,
#              'auc_roc_weighted': 1.0,
#              'f1_macro': 1.0,
#              'f1_micro': 1.0,
#              'precision_macro': 1.0,
#              'precision_micro': 1.0,
#              'recall_macro': 1.0,
#              'recall_micro': 1.0,
#              'top1_acc_macro': 1.0,
#              'top1_acc_micro': 1.0,
#              'top1_acc_weighted': 1.0},
#  'num_classes': 10,
#  'package version': '0.1.0',
#  'task': 'Image Classification'}