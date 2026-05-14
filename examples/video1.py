# pip install transformers 
# pip install decord 
import aicard_eval

import torch
import tempfile
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import numpy as np
from datasets import load_dataset
from pprint import pprint

# Step 1: Load model
device = "cuda"
model_id = "nateraw/videomae-base-finetuned-ucf101"
processor = VideoMAEImageProcessor.from_pretrained(model_id)
model = VideoMAEForVideoClassification.from_pretrained(model_id).to(device)

# Step 2: Load dataset
dataset = load_dataset("aisuko/ucf101-subset", split='train') # first 10 classes of ucf101
split = []
labels = []
for data in dataset:
    split.append(data['__key__'].removeprefix('UCF101_subset/').split('/')[0])
    labels.append(model.config.label2id[data['__key__'].removeprefix(f'UCF101_subset/{split[-1]}/').split('/')[0]])
  
dataset = dataset.add_column("split", split)
dataset = dataset.add_column("label", labels)
dataset_test = dataset.filter(lambda x: x["split"] == "test")

# Step 3: Define pipeline
def pipeline(data):
    # prepare input
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as f:
        f.write(data['avi'][0])
        f.flush()
        vr = VideoReader(f.name, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, 16).astype(int) # 16 frames
    frames = vr.get_batch(indices).asnumpy()
    inputs = processor(list(frames), return_tensors="pt").to(device)
    
    # inference and logits (first 10 classes)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = outputs.logits.cpu().tolist()[0][0:10]
    
    # probabilities
    logits = np.array(pred_idx)
    exp_logits = np.exp(logits - np.max(logits)) # subtract max for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)

    return [probabilities.tolist()]

# Step 4: Run evaluation
metrics = aicard_eval.evaluate(
    data=dataset_test,
    pipeline=pipeline,
    task=aicard_eval.tasks.vision.video_classification)

pprint(metrics)
# {'batch_size': 1,
#  'datetime': '2026-Jan-12 13:39',
#  'energy_consumption': '0.0001625975238753 kWh, ',
#  'execution_time': 'inference: 5.82s, metrics: 12.48ms',
#  'hardware': "CPU: ['AMD Ryzen 7 7800X3D 8-Core Processor/1 device(s), "
#              "TDP:120.0'], RAM: 15.62 GB, GPU: ['NVIDIA GeForce RTX 4070 SUPER "
#              "1 device(s)'] CUDA: | NVIDIA-SMI 590.44.01              Driver "
#              'Version: 591.44         CUDA Version: 13.1     |',
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
#  'package_version': '0.1.2',
#  'task': 'Video Classification'}