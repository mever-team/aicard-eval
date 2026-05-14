# pip install transformers
# install pytorch based on you CUDA version
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
#  'cpu_model': 'AMD Ryzen 7 7800X3D 8-Core Processor',
#  'date': '2026-May-14',
#  'datetime': '2026-May-14 18:19',
#  'emissions': '208.987 mg',
#  'energy_consumption': '620.925 mWh',
#  'execution_time': 'inference: 30.62s, metrics: 66.70ms',
#  'gpu_model': '1 x NVIDIA GeForce RTX 4070 SUPER',
#  'metrics': {'auc_roc_macro': 0.929,
#              'auc_roc_weighted': 0.91,
#              'f1_macro': 0.466,
#              'f1_micro': 0.574,
#              'max_dfnr': 0.201,
#              'max_dfpr': 0.034,
#              'min_prule': 0.655,
#              'precision_macro': 0.509,
#              'precision_micro': 0.574,
#              'recall_macro': 0.465,
#              'recall_micro': 0.574,
#              'top1_acc_macro': 0.574,
#              'top1_acc_micro': 0.574,
#              'top1_acc_weighted': 0.574},
#  'num_classes': 28,
#  'package_version': '0.1.4',
#  'ram_total_size': 15.621414184570312,
#  'task': 'Text Classification'}