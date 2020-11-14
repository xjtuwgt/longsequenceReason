import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import LongformerTokenizer

import logging
import os
import pandas as pd
from time import time
from torch.utils.data import DataLoader
from modelTrain.QATrainFunction import get_date_time, read_train_dev_data_frame, log_metrics
from modelTrain.QATrainFunction import test_all_steps, get_dev_data_loader, get_model
from multihopUtils.gpu_utils import set_seeds
from multihopQA.hotpotQAdataloader import HotpotTrainDataset
from multihopUtils.longformerQAUtils import LongformerQATensorizer
from torch import Tensor as T

def get_par_train_data_loader(rank, args) -> (DataLoader, DistributedSampler, int):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.train_data_name)
    data_size = data_frame.shape[0]
    if args.train_data_filtered == 1:
        data_frame = data_frame[data_frame['level'] != 'easy']
        logging.info('Filtered data by removing easy case {} to {}'.format(data_size, data_frame.shape[0]))
    elif args.train_data_filtered == 2:
        data_frame = data_frame[data_frame['level'] == 'hard']
        logging.info('Filtered data by removing easy and medium case {} to {}'.format(data_size, data_frame.shape[0]))
    else:
        logging.info('Using all training data {}'.format(data_size))
    data_size = data_frame.shape[0]

    num_replicas = args.world_size
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataset = HotpotTrainDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer,
                      max_sent_num=args.max_sent_num)
    batch_size = args.batch_size // num_replicas
    logging.info('Each node batch size = {}'.format(batch_size))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, rank=rank, num_replicas=num_replicas)
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=max(1, args.cpu_num // 2),
                                  collate_fn=HotpotTrainDataset.collate_fn,
                                  shuffle=False,
                                  pin_memory=True,
                                  sampler=train_sampler)
    return train_dataloader, train_sampler, data_size

def init_process(rank, word_size, backend='nccl'):
    """ Initialize the distributed environment. """
    """
    Backend: communication backend to be used. Options available : Gloo, NCCL, MPI.
    NCCL is suitable for GPU training while Gloo is suited more for CPU training
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch_dist.init_process_group(rank=rank, world_size=word_size, backend=backend)

def main_worker(rank, world_size, args):
    start_time = time()
    init_process(rank=rank, word_size=world_size)
    logging.info('Rank {}/{} training process initialized.'.format(rank, world_size))
    ##++++++++++++++++++++++++++++
    set_seeds(seed=args.rand_seed)
    ##++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++
    logging.info(f"Rank {rank}/{world_size} training process passed data download barrier.\n")
    device = torch.device("cuda:{}".format(rank))
    model = get_model(args=args).to(device)
    ddp_model = DDP(module=model, device_ids=[rank], output_device=device, find_unused_parameters=True)
    train_data_loader, train_sampler, train_size = get_par_train_data_loader(rank=rank, args=args)
    dev_data_loader, _ = get_dev_data_loader(args=args)
    logging.info('Start model training...')
    logging.info('*' * 75)
    ##++++++++++++++++++++++++++++
    if rank == 0:
        torch_dist.barrier()
    ##++++++++++++++++++++++++++++
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate * world_size,
                                 weight_decay=args.weight_decay)
    training_logs = []
    step = 0
    for epoch in range(1, args.epoch + 1):
        train_sampler.set_epoch(epoch)
        for batch_idx, sample in enumerate(train_data_loader):
            log = train_single_step(model=ddp_model, optimizer=optimizer, device=device, train_sample=sample, args=args)
            step = step + 1
            training_logs.append(log)
            ##+++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch + 1, batch_idx + 1,
                                                                                         time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and rank == 0:
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = test_all_steps(model=ddp_model, test_data_loader=dev_data_loader, device=device, args=args)
                answer_type_acc = metric_dict['answer_type_acc']
                eval_metric = answer_type_acc
                for key, metrics in metric_dict.items():
                    if key.endswith('metrics'):
                        logging.info('Metrics = {}'.format(key))
                        logging.info('*' * 75)
                        log_metrics('Valid', step, metrics)
                        logging.info('*' * 75)
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                dev_data_frame = metric_dict['res_dataframe']
                date_time_str = get_date_time()
                dev_result_name = os.path.join(args.save_path,
                                               date_time_str + str(step) + "_acc_" + answer_type_acc + '.json')
                dev_data_frame.to_json(dev_result_name, orient='records')
                logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Training completed...')
    if rank == 0:
        logging.info('Evaluating on Valid Dataset with final model...')
        metric_dict = test_all_steps(model=ddp_model, test_data_loader=dev_data_loader, device=device, args=args)
        answer_type_acc = metric_dict['answer_type_acc']
        for key, metrics in metric_dict.items():
            if key.endswith('metrics'):
                logging.info('Metrics = {}'.format(key))
                logging.info('*' * 75)
                log_metrics('Valid', 'all_steps', metrics)
                logging.info('*' * 75)
        logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
        logging.info('*' * 75)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        dev_data_frame = metric_dict['res_dataframe']
        date_time_str = get_date_time()
        dev_result_name = os.path.join(args.save_path,
                                       date_time_str + str(step) + "_acc_" + answer_type_acc + '.json')
        dev_data_frame.to_json(dev_result_name, orient='records')
        logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
        logging.info('*' * 75)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        logging.info('Evaluation completed...')

def train_single_step(model, optimizer, train_sample, device, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    model.train()
    model.zero_grad()
    if args.cuda:
        sample = dict()
        for key, value in train_sample.items():
            sample[key] = value.to(device)
    else:
        sample = train_sample
    loss_output = model(sample)
    yn_loss, span_loss, supp_doc_loss, supp_sent_loss = loss_output['yn_loss'], \
                                                        loss_output['span_loss'], \
                                                        loss_output['doc_loss'], loss_output['sent_loss']
    supp_doc_pair_loss = loss_output['doc_pair_loss']
    if args.do_retrieval:
        loss = supp_doc_loss + supp_sent_loss + supp_doc_pair_loss * args.pair_score_weight
    else:
        loss = supp_doc_loss + supp_sent_loss + span_loss * args.span_weight + yn_loss + supp_doc_pair_loss * args.pair_score_weight

    optimizer.zero_grad()
    loss.sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'al_loss': loss.sum().item(),
        'an_loss': span_loss.sum().item(),
        'sd_loss': supp_doc_loss.sum().item(),
        'sp_loss': supp_doc_pair_loss.sum().item(),
        'ss_loss': supp_sent_loss.sum().item(),
        'yn_loss': yn_loss.sum().item()
    }
    return log

def run_train_and_dev(args):
    if args.world_size > 1:
        ngpu_per_node = torch.cuda.device_count()
        if args.world_size > ngpu_per_node:
            args.world_size = ngpu_per_node
        mp.spawn(fn=main_worker, nprocs=args.world_size, args=(args.world_size, args))
    else:
        main_worker(rank=0, world_size=1, args=args)