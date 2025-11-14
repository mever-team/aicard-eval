from datetime import datetime
import time
import inspect

import aicard_eval
from .utils import (human_readable_time,
                              get_hardware_info,
                              is_path,
                              read_data,
                              convert_to_datasets,
                              anns_to_datasets,
                              check_validity_of_target)


def autocall(metric, **kwargs):
    args = set(inspect.signature(metric).parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in args}
    try: return metric(**kwargs)
    except TypeError as e:
        print(e)
        return None

def evaluate(
    data: "path or data",
    pipeline: callable,
    task: aicard_eval.tasks.Task,
    target_column:str|None=None,
    num_classes:int|None=None,  # in case the preds have more classes than target
    batch_size:int=1,
    anns: list[list[dict]]|list[dict]|None=None,
) -> dict:
    if anns is None:
        anns = [None]
    if is_path(data):
        data = read_data(data)
    data = convert_to_datasets(data)
    data = data.batch(batch_size)
    anns = anns_to_datasets(anns)
    anns = anns.batch(batch_size)

    target_column = check_validity_of_target(anns[0] if len(anns.features) else data[0], task, target_column)
    out_sample = pipeline(data[0])
    task.assert_output_type(out_sample[0])

    preds = []
    start = time.time()
    for batch in data:
        preds.extend(pipeline(batch))
    pipe_execution_time = time.time() - start
    kwargs = task.parameters(
        data=data,
        preds=preds,
        target_column=target_column,
        num_classes=num_classes,
        anns=anns
    )

    if 'num_classes' in kwargs and kwargs['num_classes'] == 2: task.metrics.append(aicard_eval.metrics.precision_recall_curves)
    start = time.time()
    metrics = {metric.__name__: autocall(metric, **kwargs) for metric in task.metrics}
    metrics_execution_time = time.time() - start
    # metrics = {k: float(v) for k,v in metrics.items() if v is not None}

    caller_path = inspect.stack()[1].filename
    with open(caller_path, 'r') as f:
        caller_content = f.read()

    out = {
        'package version': aicard_eval.__version__,
        'datetime': datetime.now().strftime('%Y-%b-%d %H:%M'),
        'task':task.name ,
        'metrics': metrics,
        'batch_size': batch_size,
        'code': caller_content,
        'hardware': get_hardware_info(),
        'execution_time': f'inference: {human_readable_time(pipe_execution_time)}, metrics: {human_readable_time(metrics_execution_time)}',
        }

    if 'num_classes' in kwargs:out['num_classes'] = kwargs['num_classes']

    return out
