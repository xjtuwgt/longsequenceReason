import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import logging
import os
import pandas as pd
from time import time
import torch
from torch.utils.data import DataLoader
from multihopQA.hotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset
from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder, get_hotpotqa_longformer_tokenizer
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from datetime import date, datetime
########################################################################################################################
MASK_VALUE = -1e9
########################################################################################################################

def read_train_dev_data_frame(file_path, json_fileName):
    start_time = time()
    data_frame = pd.read_json(os.path.join(file_path, json_fileName), orient='records')
    logging.info('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {} at step {}: {:.4f}'.format(mode, metric, step, metrics[metric]))

def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str

def get_train_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.train_data_name)
    batch_size = args.batch_size
    #####################################################
    training_data_shuffle = args.training_shuffle == 1
    #####################################################
    data_size = data_frame.shape[0]
    if args.train_data_filtered == 1:
        data_frame = data_frame[data_frame['level'] != 'easy']
        logging.info('Filtered data by removing easy case {} to {}'.format(data_size, data_frame.shape[0]))
    elif args.train_data_filtered == 2:
        data_frame = data_frame[data_frame['level'] == 'hard']
        logging.info(
            'Filtered data by removing easy and medium case {} to {}'.format(data_size, data_frame.shape[0]))
    else:
        logging.info('Using all training data {}'.format(data_size))

    data_size = data_frame.shape[0]
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotTrainDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, max_doc_num=args.max_doc_num,
                           max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type, training_shuffle=training_data_shuffle),
        batch_size=batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotTrainDataset.collate_fn
    )
    return dataloader, data_size


def get_dev_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
    batch_size = args.test_batch_size
    data_size = data_frame.shape[0]
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, max_doc_num=args.max_doc_num,
                         hotpot_tensorizer=hotpot_tensorizer, max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDevDataset.collate_fn
    )
    return dataloader, data_size

def get_model(args):
    start_time = time()
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name)
    longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                 hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                 seq_project=args.seq_project)
    longEncoder.resize_token_embeddings(len(tokenizer))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.frozen_layer_num > 0:
        modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:args.frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
    logging.info('Loading encoder takes {:.4f}'.format(time() - start_time))
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels, args=args)
    logging.info('Constructing reasonModel completes in {:.4f}'.format(time() - start_time))
    return model

def get_check_point(args):
    start_time = time()
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name, do_lower_case=True)
    longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                 hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                 seq_project=args.seq_project)
    longEncoder.resize_token_embeddings(len(tokenizer))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.frozen_layer_num > 0:
        modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:args.frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
    logging.info('Loading encoder takes {:.4f}'.format(time() - start_time))
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels, args=args)
    logging.info('Constructing reasonModel completes in {:.4f}'.format(time() - start_time))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model_path = args.save_path
    model_file_name = args.init_checkpoint
    hotpot_qa_model_name = os.path.join(model_path, model_file_name)
    # model, optimizer, _, _, _ = load_check_point(model=model, optimizer=optimizer, PATH=hotpot_qa_model_name)
    return model, optimizer