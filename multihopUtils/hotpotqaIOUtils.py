import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import os
import pandas as pd
from pandas import DataFrame
import torch
import json
from torch.optim import Adam
from time import time

hotpot_path = '../data/hotpotqa/'
hotpot_path = os.path.abspath(hotpot_path)
print('Orignal data path = {}'.format(hotpot_path))
hotpot_train_data = 'hotpot_train_v1.1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_dev_fullwiki = 'hotpot_dev_fullwiki_v1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_test_fullwiki = 'hotpot_test_fullwiki_v1.json'  # _id; question; context
hotpot_dev_distractor = 'hotpot_dev_distractor_v1.json'  # _id;answer;question;supporting_facts;context;type;level
gold_hotpot_dev_distractor = 'gold_hotpot_dev_distractor_v1.json'

def loadWikiData(PATH, json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(os.path.join(PATH, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def HOTPOT_TrainData(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_train_data)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_FullWiki(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_dev_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_Distractor(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_dev_distractor)
    column_names = [col for col in data.columns]
    return data, column_names

def GOLD_HOTPOT_DevData_Distractor(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=gold_hotpot_dev_distractor)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_Test_FullWiki(path=hotpot_path):
    data = loadWikiData(PATH=path, json_fileName=hotpot_test_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def save_data_frame_to_json(df: DataFrame, file_name: str):
    df.to_json(file_name, orient='records')
    print('Save {} data in json file'.format(df.shape))

###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def save_check_point(model, optimizer: Adam, loss, eval_metric, step, args):
    argparse_dict = vars(args)
    if (loss is not None) and (eval_metric is not None):
        save_model_name = '{}_{}_{}'.format(step, loss, eval_metric)
    else:
        save_model_name = '{}'.format(step)
    with open(os.path.join(args.save_path, 'config_' + save_model_name + '.json'), 'w') as fjson: ## saving model parameters
        json.dump(argparse_dict, fjson)
    model_to_save = model
    save_path = os.path.join(args.save_path, 'model_' + save_model_name + '.pt')
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    torch.save({
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'eval': eval_metric
    }, save_path)
    return save_path

def load_check_point_for_train(model, optimizer: Adam, PATH: str):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        checkpoint = torch.load(PATH, device)
    else:
        checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    eval_metric = checkpoint['eval']
    return model, optimizer, step, loss, eval_metric

def load_check_point(model, model_name: str, PATH: str):
    model_path_name = os.path.join(PATH, model_name)
    model = load_model(model=model, model_path=model_path_name)
    return model

def load_model(model, model_path: str):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        checkpoint = torch.load(model_path, device)
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model