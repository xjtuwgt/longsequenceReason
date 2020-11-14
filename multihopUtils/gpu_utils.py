import subprocess
from io import StringIO
import torch
import pandas as pd
import numpy as np
import random
import os

def get_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx]['memory_used']
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx, used_memory

def set_free_cuda():
    free_gpu_id, used_memory = get_free_gpu()
    device = torch.device('cuda:'+str(free_gpu_id))
    torch.cuda.set_device(device=device)
    return [free_gpu_id], used_memory

def get_multi_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx]['memory_used']
    if used_memory < 1000:
        used_memory = 1000
    free_idxs = []
    for idx, row in gpu_df.iterrows():
        if row['memory_used'] <= used_memory:
            free_idxs.append(idx)
    print('Returning GPU {} with smaller than {} free MiB'.format(free_idxs, gpu_df.iloc[idx]['memory.free']))
    return free_idxs, used_memory

def set_multi_free_cuda():
    free_gpu_ids, used_memory = get_multi_free_gpu()
    aa = []
    for i in free_gpu_ids:
        device = torch.device("cuda:{}".format(i))
        aa.append(torch.rand(1).to(device))  # a place holder
    return free_gpu_ids, used_memory

def gpu_setting(num_gpu=1):
    if num_gpu > 1:
        return set_multi_free_cuda()
    else:
        return set_free_cuda()

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True