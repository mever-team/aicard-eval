# AICard-Eval

This is a package created under the AI-CODE and it's part of the transparency services for AI model cards. Its purpose is to provide a single tool for evaluating AI models with performance and bias metrics. The output is standardized and meant (but not restricted) to be used as an import for aicard package.

## ⚡ Quickstart

To install use:

```bash
conda create -n aicard-eval python=3.11
conta activate aicard-eval
pip install aicard-eval
```
or if you clone this repo
```bash
pip install -e .
```
Follow the script bellow. The aicard-eval will choose the correct metrics corresponding to your case. For more examples see the examples/ folder. 

You can use datasets and models from service providers e.g. huggingface or you can use your local models and datasets. Supported datasets types are: .csv, .tsv, .json, .jsonl, .xml, .yml, .yaml, .parquet, .feather, .pickle and supported image types are .jpg, .jpeg, .png, .gif, .bmp, .tiff, .tif

```python
import aicard_eval
from datasets import load_dataset
from transformers import pipeline

# 1) Load your model
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# 2) Load your dataset
dataset = load_dataset("google-research-datasets/go_emotions", split='test')
class_names = dataset.features["labels"].feature.names


# 3) Define a function to handle the dataset
def pipeline(data):
        sentences = [text for text in data['text']]
        model_outputs = classifier(sentences)
        out = []
        for sample in model_outputs:
            flat = {d['label']: d['score'] for d in sample}
            out.append([flat[name] for name in class_names])
        return out

# 4) call the aicard-eval evaluate function
metrics = aicard_eval.evaluate(
    data=dataset,
    pipeline=pipeline,
    task=aicard_eval.tasks.nlp.text_classification,
    batch_size=32)

print(metrics)
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
```

You can also upload your metrics to our model card database. This will create an html formated report of your inference run and store it in your own model card.

```python
from aicard_eval.utils import upload
upload(username = USER_NAME, password = PASSWORD, metrics = metrics, card_id = 116)
```

## 💡 Pipeline Instructions

The pipeline funtion is the inference loop that the evaluate function calls to generate the predictions of the model. It is completely abstract which means it can contain whatever the user wants. There are only two rules to follow to construct the pipeline:

1) It must have a single function parameter `def pipeline(data)`
2) It must return a specific format depending on the task.

The package supports several formats for each task but until they are thoroughly tested here is a list you can follow:
| Task | Return Format | Example |
|----------|----------|----------|
| Binary Classification    | list [ int ] | [ 0,1,0,0 ] |
| Multi-class Classification    | list[ list[ float ] ] |  [ [0.654, 0.125, 0.471], [0.268, 0.659, 0.073]] |
| Multi-label Classification    | list[ list[ int ] ] |  [ [ 2 ],[ 9,3 ],[ 3,0,1 ],[ 0 ] ] |
| Object Detection    | list[ dict ] | [{<br>"boxes": [ [ 25, 27, 37, 54 ], [ 119, 111, 40, 67 ] ],<br>"labels": [ 0, 1 ],<br>"scores": [ .88, .70 ]<br>},<br>{<br>"boxes": [ [ 64, 111, 64, 58 ] ],<br>"labels": [ 0 ],<br>"scores": [ .71 ]`<br>}] |

<br>

On the other hand `data` is basically the dataset the user imported split into batches of size `batch_size`. A loop will call the pipeline function until all batches are processed by it. The `data` is a dictionary of lists `dict[str, list]`. For example if we import a .csv:

```
name, age
Alice, 30
Bob, 25
Charlie, 35
```
with `batch_size=3` then 
```
>>> data['name']
['Alice', 'Bob', 'Charlie']
>>> data['age'][0]
30
```
## 🔌 Energy Consumption
If you want to use the package as an energy consumption tracker, use the bellow instructions:
On your python environment install the package 
```bash
pip install aicard-eval
```
and measure the energy consumption like the example bellow
```python
from aicard_eval.emissions import CarbonTrack

# initialize the tracker
emission_tracker = CarbonTrack()
# initialize run
run_name = 'my_run'
emission_tracker.start(run_name)
# execute some code
some_code()
# stop the tracker and obtain the output
emissions = emission_tracker.stop(run_name)
emissions_out = {k: emissions[k] for k in ['energy_consumed', 'emissions', 'cpu_model', 'gpu_model', 'ram_total_size']}
print(emissions_out)
```
If you are using this on a server, make sure you have only one instance of `CarbonTrack()` defined. You can initialize multiple runs with the same instance and truck multiple runs at the same time.  
> [!NOTE]
> place holder
