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
dataset = load_dataset("aisuko/ucf101-subset", split='train')
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
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=True) as f:
        f.write(data['avi'][0])
        f.flush()
        vr = VideoReader(f.name, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, 16).astype(int) # 16 frames
    frames = vr.get_batch(indices).asnumpy()
    
    inputs = processor(list(frames), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_idx = outputs.logits.argmax(-1).item()
    return [pred_idx]

# Step 4: Run evaluation
metrics = aicard_eval.evaluate(
    data=dataset_test,
    pipeline=pipeline,
    task=aicard_eval.tasks.vision.video_classification)

pprint(metrics)