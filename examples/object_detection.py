import evaluation
import numpy as np


def preds(data):
 return [{
    "boxes": np.array([[296.55, 93.96, 314.97, 152.79],
                       [298.55, 98.96, 314.97, 151.79]], dtype=np.float32),
    "labels": np.array([4, 5], dtype=np.int64),
    "scores": np.array([0.9, 0.8], dtype=np.float32)}]

metrics = evaluation.evaluate(
    data={
        "boxes": [[[300.00, 100.00, 315.00, 150.00],[300.00, 100.00, 315.00, 150.00]]],
        "labels": [[4,5]]},
    pipeline=preds,
    task=evaluation.tasks.vision.object_detection)

print(metrics['metrics'])