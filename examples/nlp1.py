import evaluation
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

classifier = pipeline(
    "text-classification",
    model='vectara/hallucination_evaluation_model',
    tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
    trust_remote_code=True,
    device = 0
)
dataset = load_dataset("lytang/LLM-AggreFact")
data_test = dataset['test']

def pipeline(data):
    claim = [sample[:256] for sample in data['claim']]
    doc = [sample[:256] for sample in data['doc']]
    pairs = [(c, d) for c, d in zip(claim, doc)]
    prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
    input_pairs = [prompt.format(text1=pair[0], text2=pair[1]) for pair in pairs]
    full_scores = classifier(input_pairs, top_k=None)
    simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
    return simple_scores


metrics = evaluation.evaluate(
    data=data_test.select(range(200)),
    pipeline=pipeline,
    task=evaluation.tasks.nlp.text_classification,
    batch_size=4)

print(metrics)



