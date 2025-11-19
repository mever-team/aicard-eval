from PIL import Image
import numpy as np
import io
import aicard_eval
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from datasets import load_dataset


dataset = load_dataset("rishitdagli/cppe-5", split='test').select(range(5))

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
    
    for results in resultss:
        print('')
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
            )
    
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
    batch_size=3,
    box_format='xywh')

print(metrics)
