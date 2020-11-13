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
from pandas import DataFrame
from multihopUtils.gpu_utils import set_seeds
from multihopQA.hotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset
from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from datetime import date, datetime
from torch import Tensor as T

def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str

def read_train_dev_data_frame(file_path, json_fileName) -> DataFrame:
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

def load_train_data(rank, args) -> (DataLoader, DistributedSampler, int):
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

def load_dev_data(args) -> DataLoader:
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataset = HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer,
                      max_sent_num=args.max_sent_num)
    dev_dataloader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, num_workers=max(1, args.cpu_num // 2),
                                  collate_fn=HotpotDevDataset.collate_fn,
                                  shuffle=False,
                                  pin_memory=True)
    return dev_dataloader

def init_process(rank, word_size, backend='nccl'):
    """ Initialize the distributed environment. """
    """
    Backend: communication backend to be used. Options available : Gloo, NCCL, MPI.
    NCCL is suitable for GPU training while Gloo is suited more for CPU training
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch_dist.init_process_group(rank=rank, world_size=word_size, backend=backend)

def get_model(args):
    start_time = time()
    longformer = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                attn_dropout=args.attn_drop, hidden_dropout=args.input_drop,
                                                 seq_project=args.seq_project)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.frozen_layer_num > 0:
        modules = [longformer.embeddings, *longformer.encoder.layer[:args.frozen_layer_num]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
    logging.info('Loading encoder takes {:.4f}'.format(time() - start_time))
    model = LongformerHotPotQAModel(longformer=longformer, num_labels=args.num_labels, args=args)
    logging.info('Constructing model completes in {:.4f}'.format(time() - start_time))
    return model

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
    train_data_loader, train_sampler, train_size = load_train_data(rank=rank, args=args)
    dev_data_loader = load_dev_data(args=args)
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
    # for epoch in range(1, args.epoch + 1):
    #     train_sampler.set_epoch(epoch)
    #     for batch_idx, sample in enumerate(train_data_loader):
    #         log = train_single_step(model=ddp_model, optimizer=optimizer, device=device, train_sample=sample, args=args)
    #         step = step + 1
    #         training_logs.append(log)
    #         ##+++++++++++++++++++++++++++++++++++++++++++++++
    #         if step % args.log_steps == 0:
    #             metrics = {}
    #             for metric in training_logs[0].keys():
    #                 metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
    #             log_metrics('Training average', step, metrics)
    #             train_loss = metrics['al_loss']
    #             logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch + 1, batch_idx + 1,
    #                                                                                      time() - start_time))
    #             training_logs = []
    #
    #         if args.do_valid and step % args.valid_steps == 0 and rank == 0:
    #             logging.info('Evaluating on Valid Dataset...')
    #             metric_dict = test_all_steps(model=ddp_model, test_data_loader=dev_data_loader, device=device, args=args)
    #             answer_type_acc = metric_dict['answer_type_acc']
    #             eval_metric = answer_type_acc
    #             for key, metrics in metric_dict.items():
    #                 if key.endswith('metrics'):
    #                     logging.info('Metrics = {}'.format(key))
    #                     logging.info('*' * 75)
    #                     log_metrics('Valid', step, metrics)
    #                     logging.info('*' * 75)
    #             logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
    #             logging.info('*' * 75)
    #             ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    #             dev_data_frame = metric_dict['res_dataframe']
    #             date_time_str = get_date_time()
    #             dev_result_name = os.path.join(args.save_path,
    #                                            date_time_str + str(step) + "_acc_" + answer_type_acc + '.json')
    #             dev_data_frame.to_json(dev_result_name, orient='records')
    #             logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
    #             logging.info('*' * 75)
    #             ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    # logging.info('Training completed...')
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


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++Test steps++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def test_all_steps(model, test_data_loader, device, args):
    '''
            Evaluate the model on test or valid datasets
    '''
    model.eval()
    ###########################################################
    start_time = time()
    test_dataset = test_data_loader
    doc_logs, sent_logs = [], []
    step = 0
    N = 0
    total_steps = len(test_dataset)
    # **********************************************************
    support_doc_pred_results, support_doc_true_results, support_doc_score_results = [], [], []
    support_sent_pred_results, support_sent_true_results, support_sent_score_results = [], [], []
    answer_type_pred_results, answer_type_true_results = [], []
    span_pred_start_results, span_true_start_results = [], []
    span_pred_end_results, span_true_end_results = [], []
    encode_id_results = []
    correct_answer_num = 0
    # **********************************************************
    with torch.no_grad():
        for test_sample in test_dataset:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    sample[key] = value.to(device)
            else:
                sample = test_sample
            output = model(sample)
            N = N + sample['doc_labels'].shape[0]
            eval_res = metric_computation(output_scores=output, sample=sample, args=args)
            # +++++++++++++++++++
            correct_yn, yn_predicted_labels, yn_true_labels = eval_res['answer_type']
            correct_answer_num += correct_yn
            answer_type_pred_results += yn_predicted_labels
            answer_type_true_results += yn_true_labels
            # +++++++++++++++++++
            span_start, span_end = eval_res['answer_span']
            span_predicted_start, span_true_start = span_start
            span_predicted_end, span_true_end = span_end
            span_pred_start_results += span_predicted_start
            span_pred_end_results += span_predicted_end
            span_true_start_results += span_true_start
            span_true_end_results += span_true_end
            # +++++++++++++++++++
            encode_ids = eval_res['encode_ids']
            encode_id_results += encode_ids
            # +++++++++++++++++++
            doc_metric_logs, doc_pred_res = eval_res['supp_doc']
            doc_logs += doc_metric_logs
            doc_predicted_labels, doc_true_labels, doc_score_list = doc_pred_res
            support_doc_pred_results += doc_predicted_labels
            support_doc_true_results += doc_true_labels
            support_doc_score_results += doc_score_list
            # +++++++++++++++++++
            sent_metric_logs, sent_pred_res = eval_res['supp_sent']
            sent_logs += sent_metric_logs
            sent_predicted_labels, sent_true_labels, sent_score_list = sent_pred_res
            support_sent_pred_results += sent_predicted_labels
            support_sent_true_results += sent_true_labels
            support_sent_score_results += sent_score_list
            ##-------------------------------------------------------------------
            # ******************************************

            step += 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    doc_metrics, sent_metrics = {}, {}
    for metric in doc_logs[0].keys():
        doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
    for metric in sent_logs[0].keys():
        sent_metrics[metric] = sum([log[metric] for log in sent_logs]) / len(sent_logs)
    ##=================================================
    answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
    result_dict = {'aty_pred': answer_type_pred_results, 'aty_true': answer_type_true_results,
                   'sd_pred': support_doc_pred_results, 'sd_true': support_doc_true_results,
                   'ss_pred': support_sent_pred_results, 'ss_true': support_sent_true_results,
                   'sps_pred': span_pred_start_results, 'spe_pred': span_pred_end_results,
                   'sps_true': span_true_start_results, 'spe_true': span_true_end_results,
                   'sd_score': support_doc_score_results, 'ss_score': support_sent_score_results,
                   'encode_ids': encode_id_results}
    res_data_frame = DataFrame(result_dict)
    ##=================================================
    return {'supp_doc_metrics': doc_metrics, 'supp_sent_metrics': sent_metrics,
            'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}


def metric_computation(output_scores: dict, sample: dict, args):
    # =========Answer type prediction==========================
    yn_scores = output_scores['yn_score']
    yn_true_labels = sample['yes_no']
    if len(yn_true_labels.shape) > 1:
        yn_true_labels = yn_true_labels.squeeze(dim=-1)
    yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
    correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
    yn_predicted_labels = yn_predicted_labels.detach().tolist()
    yn_true_labels = yn_true_labels.detach().tolist()
    # =========Answer type prediction==========================
    # =========Answer span prediction==========================
    start_logits, end_logits = output_scores['span_score']
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_logits = start_logits.masked_fill(sample['ctx_attn_mask'] == 0, -1e11)
    end_logits = end_logits.masked_fill(sample['ctx_attn_mask'] == 0, -1e11)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    answer_start_positions, answer_end_positions = sample['ans_start'], sample['ans_end']
    if len(answer_start_positions.shape) > 1:
        answer_start_positions = answer_start_positions.squeeze(dim=-1)
    if len(answer_end_positions.shape) > 1:
        answer_end_positions = answer_end_positions.squeeze(dim=-1)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    predicted_span_start = torch.argmax(start_logits, dim=-1)
    predicted_span_end = torch.argmax(end_logits, dim=-1)
    predicted_span_start = predicted_span_start.detach().tolist()
    predicted_span_end = predicted_span_end.detach().tolist()
    true_span_start = answer_start_positions.detach().tolist()
    true_span_end = answer_end_positions.detach().tolist()
    # =========Answer span prediction==========================
    # +++++++++ supp doc prediction +++++++++++++++++++++++++++
    doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
    doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
    supp_doc_scores, _ = output_scores['doc_score']
    doc_metric_logs, doc_pred_res = support_infor_evaluation(scores=supp_doc_scores, labels=doc_label, mask=doc_mask, pred_num=2, threshold=1e9)
    # +++++++++ supp doc prediction +++++++++++++++++++++++++++
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    supp_sent_scores = output_scores['sent_score']
    sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
    sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
    sent_metric_logs, sent_pred_res = support_infor_evaluation(scores=supp_sent_scores, labels=sent_label, mask=sent_mask, pred_num=2,
                                                               threshold=args.sent_threshold)
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    encode_ids = sample['ctx_encode'].detach().tolist()
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    return {'answer_type': (correct_yn, yn_predicted_labels, yn_true_labels),
            'answer_span': ((predicted_span_start, true_span_start), (predicted_span_end, true_span_end)),
            'supp_doc': (doc_metric_logs, doc_pred_res),
            'supp_sent': (sent_metric_logs, sent_pred_res),
            'encode_ids': encode_ids}

def sp_score(prediction, gold):
    cur_sp_pred = set(prediction)
    gold_sp_pred = set(gold)
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1

def support_infor_evaluation(scores: T, labels: T, mask: T, pred_num=2, threshold=0.8):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    predicted_labels = []
    true_labels = []
    score_list = []
    for idx in range(batch_size):
        score_list.append(masked_scores[idx].detach().tolist())
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        for i in range(pred_num, sample_size):
            if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
                pred_labels_i.append(pred_idxes_i[i])
        labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
        # +++++++++++++++++
        predicted_labels.append(pred_labels_i)
        true_labels.append(labels_i)
        # +++++++++++++++++
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
        logs.append({
            'sp_em': em_i,
            'sp_f1': f1_i,
            'sp_prec': prec_i,
            'sp_recall':recall_i
        })
    return logs, (predicted_labels, true_labels, score_list)
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++