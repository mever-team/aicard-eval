from datasets import load_dataset
from huggingface_hub import dataset_info
from transformers import pipeline
import evaluation

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

metrics = evaluation.evaluate(
    data=load_dataset("google-research-datasets/go_emotions", split='test'),
    pipeline=TextClassifier(),
    task=evaluation.tasks.nlp.text_classification,
    batch_size=32)

print(metrics)
