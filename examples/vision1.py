import evaluation
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import io

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


metrics = evaluation.evaluate(
    data=dataset.select(range(100)),
    pipeline=pipeline,
    task=evaluation.tasks.vision.image_classification,
    batch_size=1)

print(metrics)

