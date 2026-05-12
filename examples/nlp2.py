from datasets import load_dataset
from huggingface_hub import dataset_info
from transformers import pipeline
import aicard_eval
import pprint
import re

class TextClassifier:
    def __init__(self):
        info = dataset_info("google-research-datasets/go_emotions")
        self.class_names = info.card_data['dataset_info'][1]['features'][1]['sequence']['class_label']['names']
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def __call__(self, data):
        sentences = [text for text in data['text']]
        model_outputs = self.classifier(sentences)
        out = []
        for sample in model_outputs:
            flat = {d['label']: d['score'] for d in sample}
            out.append([flat[name] for name in self.class_names.values()])
        return out

def detect_sensitive_attributes(batch):
    # this is a toy method for demonstration purposes
    # overlapping gender attributes based on word detection (set values for O(1) lookup if those lists grow)
    categories = {
        "male": {"he", "his", "him", "himself"},
        "female": {"she", "hers", "her", "herself"},
    }
    texts = batch['text']
    results = {cat: [0]*len(texts) for cat in categories} # preallocate for speed
    for entry, text in enumerate(texts):
        for cat, associated_words in categories.items():
            for token in re.sub(r'[^a-z]', ' ', text.lower()).split():
                if len(token)<=1: continue # speedup
                if token in associated_words:
                    results[cat][entry] = 1
    return results


metrics = aicard_eval.evaluate(
    data=load_dataset("google-research-datasets/go_emotions", split='test'),
    pipeline=TextClassifier(),
    task=aicard_eval.tasks.nlp.text_classification,
    batch_size=32,
    sensitive_columns=detect_sensitive_attributes
)

pprint.pprint(metrics)
# {'batch_size': 32,
#  'datetime': '2025-Nov-21 12:13',
#  'execution_time': 'inference: 33.11s, metrics: 59.13ms',
#  'hardware': 'CPU: AMD Ryzen 7 7800X3D 8-Core Processor, RAM: 15.62 GB, CUDA: '
#              '| NVIDIA-SMI 580.102.01             Driver Version: '
#              '581.57         CUDA Version: 13.0     |',
#  'metrics': {'auc_roc_macro': 0.9286682043487104,
#              'auc_roc_weighted': 0.9099991445506153,
#              'f1_macro': 0.4661938061623436,
#              'f1_micro': 0.5741662060070021,
#              'precision_macro': 0.5090416420534856,
#              'precision_micro': 0.5741662060070021,
#              'recall_macro': 0.46497245851260965,
#              'recall_micro': 0.5741662060070021,
#              'top1_acc_macro': 0.5741662060070021,
#              'top1_acc_micro': 0.5741662060070021,
#              'top1_acc_weighted': 0.5741662060070021},
#  'num_classes': 28,
#  'package version': '0.1.0',
#  'task': 'Text Classification'}
