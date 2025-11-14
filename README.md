# AICard-Eval

This is a package created under the AI-CODE and it's part of the transparency services for AI model cards. Its purpose is to provide a single tool for evaluating AI models. The output is standardized and ment (but not restricted) to be used as an import for aicard package.

*Notice:*
*This is an alpha version. Suported cases are text and image binary, multiclass, and multilabel classifications*

## ⚡ Quickstart

Follow the script bellow. The aicard-eval will choose the correct metrics corresponding to your case. For more examples see the examples/ folder. 

You can use datasets and models from service providers e.g. huggingface or you can use your local models and datasets. Supported datasets types are: .csv, .tsv, .json, .jsonl, .xml, .yml, .yaml, .parquet, .feather, .pickle and supported image types are .jpg, .jpeg, .png, .gif, .bmp, .tiff, .tif

```python
import aicard_eval
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

# 1) Load your model
classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True,
    device = 0
)

# 2) Load your dataset
dataset = load_dataset("lytang/LLM-AggreFact")
data_test = dataset['test']

# 3) Define a function to handle the dataset
def pipeline(data):
    claim = [sample[:256] for sample in data['claim']]
    doc = [sample[:256] for sample in data['doc']]
    pairs = [(c, d) for c, d in zip(claim, doc)]
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]
    full_scores = classifier(input_pairs, top_k=None)
    simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
    return simple_scores

# 4) call the aicard-eval evaluation function
metrics = aicard_eval.evaluate(
    data=data_test.select(range(200)),
    pipeline=pipeline,
    task=aicard_eval.tasks.nlp.text_classification,
    batch_size=4)

print(metrics)
```

