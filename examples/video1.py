import aicard_eval

import torch
import os
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from decord import VideoReader, cpu
import numpy as np
from datasets import load_dataset

# Step 1: Load model
device = "cuda"
model_id = "nateraw/videomae-base-finetuned-ucf101"
processor = VideoMAEImageProcessor.from_pretrained(model_id)
model = VideoMAEForVideoClassification.from_pretrained(model_id).to(device)

# Step 2: Load dataset
dataset = load_dataset("aisuko/ucf101-subset")

# Step 3: Define pipeline
def pipeline(data):
    pass

# Step 4: Run evaluation
metrics = aicard_eval.evaluate(
    data=dataset,
    pipeline=pipeline,
    task=aicard_eval.tasks.vision.video_classification)