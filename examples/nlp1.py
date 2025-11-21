import aicard_eval
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import pprint

classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True,
    device = 0
)
dataset = load_dataset("lytang/LLM-AggreFact", split='test')

def pipeline(data):
    claim = [sample[:256] for sample in data['claim']]
    doc = [sample[:256] for sample in data['doc']]
    pairs = [(c, d) for c, d in zip(claim, doc)]
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]
    full_scores = classifier(input_pairs, top_k=None)
    simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
    return simple_scores


metrics = aicard_eval.evaluate(
    data=dataset.select(range(200)),
    pipeline=pipeline,
    task=aicard_eval.tasks.nlp.text_classification,
    batch_size=4)

pprint.pprint(metrics)
# {'batch_size': 4,
#  'datetime': '2025-Nov-21 11:59',
#  'execution_time': 'inference: 2.48s, metrics: 6.97ms',
#  'hardware': 'CPU: AMD Ryzen 7 7800X3D 8-Core Processor, RAM: 15.62 GB, CUDA: '
#              '| NVIDIA-SMI 580.102.01             Driver Version: '
#              '581.57         CUDA Version: 13.0     |',
#  'metrics': {'auc_roc_macro': 0.4065656565656566,
#              'auc_roc_weighted': 0.4065656565656566,
#              'f1_macro': 0.07546980202994673,
#              'f1_micro': 0.08,
#              'precision_macro': 0.47146739130434784,
#              'precision_micro': 0.08,
#              'precision_recall_curve': {'precision_curve': array([0.99 , 0.98994975, 0.98989899, ... , 1. , 1., 1.]),
#                                          'recall_curve': array([1. , 0.99494949, 0.98989899, ..., 0.01010101, 0.00505051, 0.]),
#                                          'thresholds': array([0.00128141, 0.00212657, 0.00242519, ..., 0.90394258, 0.91468281, 0.92545569])},
#              'recall_macro': 0.2878787878787879,
#              'recall_micro': 0.08,
#              'top1_acc_macro': 0.08,
#              'top1_acc_micro': 0.08,
#              'top1_acc_weighted': 0.08},
#  'num_classes': 2,
#  'package version': '0.1.0',
#  'task': 'Text Classification'}