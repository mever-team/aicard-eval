from datetime import datetime
import time
import inspect
import pickle
import platform
from .utils import emissionReadableFormat
from typing import Callable

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
    
def pipeline_loop(data, pipeline, cache_path):
    if cache_path:
        print(f"Loading cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["preds"], cache["execution_time"]

    preds = []
    start = time.time()
    for batch in data:
        preds.extend(pipeline(batch))
    pipe_execution_time = time.time() - start

    with open('cache.pkl', "wb") as f:
        pickle.dump({
            "preds": preds,
            "execution_time": pipe_execution_time
        }, f)

    return preds, pipe_execution_time

def evaluate(
    data: "path or data",
    pipeline: Callable,
    task: aicard_eval.tasks.Task,
    cache_path: str = None,
    sensitive_columns:list[str]|Callable|None=None,
    target_column:str|None=None,
    num_classes:int|None=None,  # in case the preds have more classes than target
    batch_size:int=1,
    anns: list[list[dict]]|list[dict]|None=None,
    box_format = None,
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

    try:
        # there are a ton of issues with eco2ai and these checks
        # are inserted to avoid invalidating the whole application
        # during new feature development
        from .emissions import Emission
        emission = Emission()
        emission.start()
    except:
        emission = None
    preds, pipe_execution_time = pipeline_loop(data, pipeline, cache_path)
    if emission is not None:
        try:
            emission.stop()
            emission = emission.pop()
        except:
            emission = None
    if emission is None:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info = line.strip().split(":")[1].strip()
        except FileNotFoundError:
            cpu_info = platform.processor() or platform.machine()
        emission = {'CPU_name': cpu_info, 'GPU_name': 'NA'}
    if isinstance(sensitive_columns, list):
        sensitive_columns = lambda batch: {column: batch[column] for column in sensitive_columns}
    kwargs = task.parameters(
        data=data,
        preds=preds,
        target_column=target_column,
        num_classes=num_classes,
        anns=anns,
        sensitive_columns=sensitive_columns
    )
    if box_format:
        kwargs['box_format'] = box_format
    if 'num_classes' in kwargs and kwargs['num_classes'] == 2:
        task.metrics.append(aicard_eval.metrics.precision_recall_curve)
    start = time.time()
    metrics = {metric.__name__: autocall(metric, **kwargs) for metric in task.metrics}
    metrics_execution_time = time.time() - start

    caller_path = inspect.stack()[1].filename
    with open(caller_path, 'r') as f:
        caller_content = f.read()

    out = {
        'package_version': aicard_eval.__version__,
        'datetime': datetime.now().strftime('%Y-%b-%d %H:%M'),
        'date': datetime.now().strftime('%Y-%b-%d'),
        'task':task.name ,
        'metrics': metrics,
        'batch_size': batch_size,
        'code': caller_content,
        'hardware': get_hardware_info(emission),
        'execution_time': f'inference: {human_readable_time(pipe_execution_time)}, metrics: {human_readable_time(metrics_execution_time)}',
        'energy_consumption': emissionReadableFormat(emission['power_consumption(kWh)'][0]) if 'power_consumption(kWh)' in emission else "NA",
    }


    if 'num_classes' in kwargs and kwargs['num_classes']: out['num_classes'] = kwargs['num_classes']

    return out
    
