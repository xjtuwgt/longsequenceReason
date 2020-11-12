import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import save_check_point, load_check_point
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
import pandas as pd
from time import time
import torch
from torch import Tensor as T
from torch.utils.data import DataLoader
from multihopQA.hotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset
from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder, get_hotpotqa_longformer_tokenizer
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from pandas import DataFrame
from datetime import date, datetime
##
MASK_VALUE = -1e9
##

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
    model, optimizer, _, _, _ = load_check_point(model=model, optimizer=optimizer, PATH=hotpot_qa_model_name)
    return model, optimizer

def training_warm_up(model, optimizer, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warm_up_steps
    start_time = time()
    step = 0
    training_logs = []
    logging.info('Starting warm up...')
    logging.info('*' * 75)
    #########
    model.train()
    model.zero_grad()
    #########
    for batch_idx, train_sample in enumerate(train_dataloader):
        log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
        step = step + 1
        training_logs.append(log)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Training average', step, metrics)
            logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, 'warm_up', batch_idx + 1,
                                                                                     time() - start_time))
            training_logs = []
        if step >= warm_up_steps:
            logging.info('Warm up completed in {:.4f} seconds'.format(time() - start_time))
            logging.info('*' * 75)
            break
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, args=args)
    logging.info('*' * 75)
    logging.info('Answer type prediction accuracy: {}'.format(metric_dict['answer_type_acc']))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metrics'):
            logging.info('{} prediction'.format(key))
            log_metrics('Valid', 'warm up', value)
        logging.info('*' * 75)

def train_all_steps(model, optimizer, train_dataloader, dev_dataloader, args):
    assert args.save_checkpoint_steps % args.valid_steps == 0
    warm_up_steps = args.warm_up_steps
    if warm_up_steps > 0:
        training_warm_up(model=model, optimizer=optimizer, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, args=args)
        logging.info('*' * 75)
        current_learning_rate = optimizer.param_groups[-1]['lr']
        # learning_rate = args.learning_rate * 0.5
        learning_rate = current_learning_rate * 0.5
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logging.info('Update learning rate from {} to {}'.format(current_learning_rate, learning_rate))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_steps, eta_min=1e-12)
    start_time = time()
    train_loss = 0.0
    eval_metric = None
    max_sent_pred_f1 = 0.0
    step = 0
    training_logs = []
    for epoch in range(1, args.epoch + 1):
        for batch_idx, train_sample in enumerate(train_dataloader):
            log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            scheduler.step()
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            step = step + 1
            training_logs.append(log)
            ##+++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.save_checkpoint_steps == 0:
                save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss, eval_metric=eval_metric, args=args)
                logging.info('Saving the mode in {}'.format(save_path))
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch, batch_idx + 1, time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, args=args)
                logging.info('*' * 75)
                answer_type_acc = metric_dict['answer_type_acc']
                eval_metric = answer_type_acc
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                sent_pred_f1 = metric_dict['supp_sent_metrics']['sp_f1']
                logging.info('*' * 75)
                # log_metrics('Valid', step, metric_dict['metrics'])
                for key, value in metric_dict.items():
                    if key.endswith('metrics'):
                        logging.info('{} prediction'.format(key))
                        log_metrics('Valid', step, value)
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                dev_data_frame = metric_dict['res_dataframe']
                date_time_str = get_date_time()
                dev_result_name = os.path.join(args.save_path,
                                               date_time_str + '_' + str(step) + "_acc_" + answer_type_acc + '.json')
                dev_data_frame.to_json(dev_result_name, orient='records')
                logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                if max_sent_pred_f1 < sent_pred_f1:
                    max_sent_pred_f1 = sent_pred_f1
                    save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss,
                                                 eval_metric=max_sent_pred_f1, args=args)
                    logging.info('Saving the mode in {} with current best metric = {:.4f}'.format(save_path, max_sent_pred_f1))
    logging.info('Training completed...')

def train_single_step(model, optimizer, train_sample, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    if args.cuda:
        sample = dict()
        for key, value in train_sample.items():
            sample[key] = value.cuda()
    else:
        sample = train_sample
    loss_output = model(sample)
    yn_loss, span_loss, supp_doc_loss, supp_sent_loss = loss_output['yn_loss'], \
                                                        loss_output['span_loss'], \
                                                        loss_output['doc_loss'], loss_output['sent_loss']
    supp_doc_pair_loss = loss_output['doc_pair_loss']
    span_loss = span_loss * args.span_weight
    supp_doc_pair_loss = supp_doc_pair_loss * args.pair_score_weight

    if args.task == 'doc':
        loss = supp_doc_loss + supp_doc_pair_loss
    elif args.task == 'doc_sent':
        loss = supp_doc_loss + supp_sent_loss + supp_doc_pair_loss
    elif args.task == 'doc_sent_ans':
        loss = supp_doc_loss + supp_sent_loss + span_loss + yn_loss + supp_doc_pair_loss
    else:
        raise ValueError('task %s not supported' % args.task)
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'al_loss': loss.mean().item(),
        'an_loss': span_loss.mean().item(),
        'sd_loss': supp_doc_loss.mean().item(),
        'pd_loss': supp_doc_pair_loss.mean().item(),
        'ss_loss': supp_sent_loss.mean().item(),
        'yn_loss': yn_loss.mean().item()
    }
    return log
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++Test steps++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def test_all_steps(model, test_data_loader, args):
    '''
            Evaluate the reasonModel on test or valid datasets
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
    support_doc_pred_results, support_doc_score_results = [], []
    support_sent_pred_results, support_sent_score_results, support_sent_doc_sent_pair_results = [], [], []
    answer_type_pred_results = []
    span_pred_start_results = []
    span_pred_end_results = []
    encode_id_results = []
    correct_answer_num = 0
    # **********************************************************
    with torch.no_grad():
        for test_sample in test_dataset:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    sample[key] = value.cuda()
            else:
                sample = test_sample
            output = model(sample)
            N = N + sample['doc_labels'].shape[0]
            eval_res = metric_computation(output_scores=output, sample=sample, args=args)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            correct_yn, yn_predicted_labels = eval_res['answer_type']
            correct_answer_num += correct_yn
            answer_type_pred_results += yn_predicted_labels
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            span_predicted_start, span_predicted_end = eval_res['answer_span']
            span_pred_start_results += span_predicted_start
            span_pred_end_results += span_predicted_end
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            encode_ids = eval_res['encode_ids']
            encode_id_results += encode_ids
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            doc_metric_logs, doc_pred_res = eval_res['supp_doc']
            doc_logs += doc_metric_logs
            doc_predicted_labels, doc_score_list = doc_pred_res
            support_doc_pred_results += doc_predicted_labels
            support_doc_score_results += doc_score_list
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_metric_logs, sent_pred_res = eval_res['supp_sent']
            sent_logs += sent_metric_logs
            sent_predicted_labels, sent_score_list, doc_sent_fact_pair = sent_pred_res
            support_sent_pred_results += sent_predicted_labels
            support_sent_score_results += sent_score_list
            support_sent_doc_sent_pair_results += doc_sent_fact_pair
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ******************************************
            step += 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    doc_metrics, sent_metrics = {}, {}
    for metric in doc_logs[0].keys():
        doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
    for metric in sent_logs[0].keys():
        sent_metrics[metric] = sum([log[metric] for log in sent_logs]) / len(sent_logs)
    ##=================================================
    answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
    result_dict = {'aty_pred': answer_type_pred_results,
                   'sd_pred': support_doc_pred_results,
                   'ss_pred': support_sent_pred_results,
                   'sps_pred': span_pred_start_results,
                   'spe_pred': span_pred_end_results,
                   'sd_score': support_doc_score_results,
                   'ss_score': support_sent_score_results,
                   'ss_ds_pair': support_sent_doc_sent_pair_results,
                   'encode_ids': encode_id_results} ## for detailed results checking
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
    # =========Answer span prediction==========================
    start_logits, end_logits = output_scores['span_score']
    predicted_span_start = torch.argmax(start_logits, dim=-1)
    predicted_span_end = torch.argmax(end_logits, dim=-1)
    predicted_span_start = predicted_span_start.detach().tolist()
    predicted_span_end = predicted_span_end.detach().tolist()
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++ supp doc prediction +++++++++++++++++++++++++++
    doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
    doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
    supp_doc_scores, _ = output_scores['doc_score']
    doc_metric_logs, doc_pred_res = support_doc_infor_evaluation(scores=supp_doc_scores, labels=doc_label, mask=doc_mask, pred_num=2)
    # +++++++++ supp doc prediction +++++++++++++++++++++++++++
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    supp_sent_scores = output_scores['sent_score']
    sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
    sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
    sent_fact_doc_idx, sent_fact_sent_idx = sample['s2d_map'], sample['sInd_map']
    sent_metric_logs, sent_pred_res = support_sent_infor_evaluation(scores=supp_sent_scores, labels=sent_label, mask=sent_mask, pred_num=2,
                                                               threshold=args.sent_threshold, doc_fact=sent_fact_doc_idx, sent_fact=sent_fact_sent_idx)
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    encode_ids = sample['ctx_encode'].detach().tolist()
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    return {'answer_type': (correct_yn, yn_predicted_labels),
            'answer_span': (predicted_span_start, predicted_span_end),
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

def support_doc_infor_evaluation(scores: T, labels: T, mask: T, pred_num=2):
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
    return logs, (predicted_labels, score_list)
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def support_sent_infor_evaluation(scores: T, labels: T, mask: T, doc_fact: T, sent_fact: T, pred_num=2, threshold=0.8):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    predicted_labels = []
    true_labels = []
    score_list = []
    doc_sent_pair_list = []
    for idx in range(batch_size):
        score_list.append(masked_scores[idx].detach().tolist())
        # ==================
        doc_fact_i = doc_fact[idx].detach().tolist()
        sent_fact_i = sent_fact[idx].detach().tolist()
        doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
        doc_sent_pair_list.append(doc_sent_pair_i)
        # ==================
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
    return logs, (predicted_labels, score_list, doc_sent_pair_list)
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def test_all_steps_hierartical(model, test_data_loader, args):
    '''
            Evaluate the reasonModel on test or valid datasets
    '''
    model.eval()
    ###########################################################
    start_time = time()
    test_dataset = test_data_loader
    doc_logs, topk_sent_logs, thresh_sent_logs = [], [], []
    step = 0
    N = 0
    total_steps = len(test_dataset)
    # **********************************************************
    support_doc_pred_results, support_doc_score_results = [], []
    topk_support_sent_pred_results, topk_support_sent_score_results, topk_support_sent_doc_sent_pair_results = [], [], []
    thresh_support_sent_pred_results, thresh_support_sent_score_results, thresh_support_sent_doc_sent_pair_results = [], [], []
    answer_type_pred_results = []
    topk_span_results = []
    threhold_span_results = []
    encode_id_results = []
    correct_answer_num = 0
    # **********************************************************
    with torch.no_grad():
        for test_sample in test_dataset:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    sample[key] = value.cuda()
            else:
                sample = test_sample
            output = model(sample)
            N = N + sample['doc_labels'].shape[0]
            eval_res = metric_computation(output_scores=output, sample=sample, args=args)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ******************************************
            step += 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the reasonModel... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    doc_metrics, topk_sent_metrics, thresh_sent_metrics = {}, {}, {}
    for metric in doc_logs[0].keys():
        doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
    for metric in topk_sent_logs[0].keys():
        topk_sent_metrics[metric] = sum([log[metric] for log in topk_sent_logs]) / len(topk_sent_logs)
    for metric in thresh_sent_logs[0].keys():
        thresh_sent_metrics[metric] = sum([log[metric] for log in thresh_sent_logs]) / len(thresh_sent_logs)
    ##=================================================
    # answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
    # result_dict = {'aty_pred': answer_type_pred_results,
    #                'sd_pred': support_doc_pred_results,
    #                'ss_pred': support_sent_pred_results,
    #                'sps_pred': span_pred_start_results,
    #                'spe_pred': span_pred_end_results,
    #                'sd_score': support_doc_score_results,
    #                'ss_score': support_sent_score_results,
    #                'ss_ds_pair': support_sent_doc_sent_pair_results,
    #                'encode_ids': encode_id_results} ## for detailed results checking
    # res_data_frame = DataFrame(result_dict)
    # ##=================================================
    # return {'supp_doc_metrics': doc_metrics, 'supp_sent_metrics': sent_metrics,
    #         'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}


def hierartical_metric_computation(output_scores: dict, sample: dict, args):
    #'yn_score', 'span_score', 'doc_score': (supp_doc_scores, supp_head_tail_scores), 'sent_score':
    yn_scores = output_scores['yn_score']
    yn_true_labels = sample['yes_no']
    if len(yn_true_labels.shape) > 1:
        yn_true_labels = yn_true_labels.squeeze(dim=-1)
    yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
    correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
    yn_predicted_labels = yn_predicted_labels.detach().tolist()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_scores, _ = output_scores['doc_score']
    doc_mask = sample['doc_lens']
    true_doc_labels = sample['doc_labels']

    #############################################################
    doc_start_position, doc_end_position = sample['doc_start'], sample['doc_end'] ## doc start and end position for answer span prediction
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent2doc_map = sample['s2d_map'] ## restore document ids
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_res_dict = supp_doc_prediction(doc_scores=doc_scores, mask=doc_mask,
                                       labels=true_doc_labels, doc_start_pos=doc_start_position,
                                       doc_end_pos=doc_end_position, sent2doc_map=sent2doc_map, top_k=3, threshold=args.doc_threshold)
    #################################################################
    sent_scores = output_scores['sent_scores']
    sentIndoc_map = sample['sInd_map']
    topk_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['top_k_sents'])
    threshold_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['threshold_sents'])
    true_sent_labels = sample['sent_labels']
    sent_lens = sample['sent_lens']

    topk_sent_res_dict = supp_sent_predictions(scores=topk_sent_scores, labels=true_sent_labels,
                                               mask=sent_lens, sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map,
                                               pred_num=2, threshold=args.sent_threshold)
    threshold_sent_res_dict = supp_sent_predictions(scores=threshold_sent_scores, labels=true_sent_labels, mask=sent_lens,
                                                    sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map, pred_num=2,
                                                    threshold=args.sent_threshold)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    topk_span_start_predictions, topk_span_end_predictions = [], []
    threshold_span_start_predictions, threshold_span_end_predictions = [], []
    span_start_scores, span_end_scores = output_scores['span_score']
    topk_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
    topk_span_start_i = torch.argmax(topk_span_start_scores, dim=-1)
    topk_span_start_predictions.append(topk_span_start_i)

    topk_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
    topk_span_end_i = torch.argmax(topk_span_end_scores, dim=-1)
    topk_span_end_predictions.append(topk_span_end_i)

    threshold_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
    threshold_span_start_i = torch.argmax(threshold_span_start_scores)
    threshold_span_start_predictions.append(threshold_span_start_i)
    threshold_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
    threshold_span_end_i = torch.argmax(threshold_span_end_scores)
    threshold_span_end_predictions.append(threshold_span_end_i)

    topk_span_start_end_pair = list(zip(topk_span_start_predictions, topk_span_end_predictions))
    threshold_span_start_end_pair = list(zip(threshold_span_start_predictions, threshold_span_end_predictions))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encode_ids = sample['ctx_encode'].detach().tolist()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    res = {'answer_type_pred': (correct_yn, yn_predicted_labels),
           'supp_doc_pred': doc_res_dict,
           'topk_sent_pred': topk_sent_res_dict,
           'threshold_sent_pred': threshold_sent_res_dict,
           'topk_span_pred': topk_span_start_end_pair,
           'threhold_span_pred': threshold_span_start_end_pair,
           'encode_id': encode_ids}
    return res

def sent_score_extraction(sent_scores: T, doc2sent_idexes: list):
    batch_size, sent_num = sent_scores.shape
    temp_scores = torch.empty(batch_size, sent_num).fill_(MASK_VALUE).to(sent_scores.device)
    for idx in range(batch_size):
        temp_scores[idx, doc2sent_idexes[idx]] = sent_scores[idx, doc2sent_idexes[idx]]
    return temp_scores

def token_score_extraction(token_scores: T, doc_start_end_pair_list: list):
    batch_size, seq_num = token_scores.shape
    temp_scores = torch.empty(batch_size, seq_num).fill_(MASK_VALUE).to(token_scores.device)
    for idx in range(batch_size):
        doc_start_end_i = doc_start_end_pair_list[idx]
        for dox_idx, doc_pair in enumerate(doc_start_end_i):
            start_i, end_i = doc_pair
            temp_scores[idx][start_i:(end_i+1)] = token_scores[idx][start_i:(end_i+1)]
    return temp_scores

def supp_sent_predictions(scores: T, labels: T, mask: T, sent2doc_map: T, sentIndoc_map: T, pred_num=2, threshold=0.8):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    predicted_labels = []
    true_labels = []
    score_list = []
    doc_sent_pair_list = []
    for idx in range(batch_size):
        score_list.append(masked_scores[idx].detach().tolist())
        # ==================
        doc_fact_i = sent2doc_map[idx].detach().tolist()
        sent_fact_i = sentIndoc_map[idx].detach().tolist()
        doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
        doc_sent_pair_list.append(doc_sent_pair_i)
        # ==================
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
    res = {'log': logs, 'prediction': predicted_labels, 'score': score_list, 'doc_sent_pair': doc_sent_pair_list}
    return res


def supp_doc_prediction(doc_scores: T, labels: T, mask: T, doc_start_pos: T, doc_end_pos: T, sent2doc_map, top_k=2, threshold=0.9):
    batch_size, doc_num = doc_scores.shape
    top_k_predictions = []
    threshold_predictions = []
    ####
    top_k_doc_start_end = []
    top_k_sent_idxes = []
    #############################################
    threshold_doc_start_end = []
    threshold_sent_idxes = []
    ############################################
    scores = torch.sigmoid(doc_scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    true_labels = []
    score_list = []
    for idx in range(batch_size):
        score_list.append(masked_scores[idx].detach().tolist())
        # ==================
        pred_idxes_i = argsort[idx].tolist()
        top_k_labels_i = pred_idxes_i[:top_k]
        threhold_labels_i = pred_idxes_i[:top_k]
        # ==================
        for i in range(top_k, doc_num):
            if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[top_k - 1]]:
                threhold_labels_i.append(pred_idxes_i[i])
        labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist()
        # +++++++++++++++++
        top_k_predictions.append(top_k_labels_i)
        threshold_predictions.append(threhold_labels_i)
        true_labels.append(labels_i)
        # +++++++++++++++++
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=top_k_labels_i, gold=labels_i)
        t_em_i, t_prec_i, t_recall_i, t_f1_i = sp_score(prediction=threhold_labels_i, gold=labels_i)
        logs.append({
            'sp_em': em_i,
            'sp_f1': f1_i,
            'sp_prec': prec_i,
            'sp_recall': recall_i,
            'threshold_sp_em': t_em_i,
            'threshold_sp_f1': t_f1_i,
            'threshold_sp_prec': t_prec_i,
            'threshold_sp_recall': t_recall_i,
        })
        ###############
        top_k_start_end_i = []
        top_k_sent_i = []
        threshold_start_end_i = []
        threshold_sent_i = []
        for topk_pre_doc_idx in top_k_labels_i:
            doc_s_i = doc_start_pos[idx][topk_pre_doc_idx].data.item()
            doc_e_i = doc_end_pos[idx][topk_pre_doc_idx].data.item()
            top_k_start_end_i.append((doc_s_i, doc_e_i))
            sent_idx_i = (sent2doc_map[idx] == topk_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
            top_k_sent_i += sent_idx_i

        for thresh_pre_doc_idx in threhold_labels_i:
            doc_s_i = doc_start_pos[idx][thresh_pre_doc_idx].data.item()
            doc_e_i = doc_end_pos[idx][thresh_pre_doc_idx].data.item()
            threshold_start_end_i.append((doc_s_i, doc_e_i))
            sent_idx_i = (sent2doc_map[idx] == thresh_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
            threshold_sent_i += sent_idx_i
        ###############
        top_k_doc_start_end.append(top_k_start_end_i)
        threshold_doc_start_end.append(threshold_start_end_i)
        top_k_sent_idxes.append(top_k_sent_i)
        threshold_sent_idxes.append(threshold_sent_i)

    res = {'log': logs,
           'top_k_doc': top_k_predictions,
           'threshold_doc': threshold_predictions,
           'true_doc': true_labels,
           'score': score_list,
           'top_k_doc2token': top_k_doc_start_end,
           'top_k_sents': top_k_sent_idxes,
           'threshold_doc2token': threshold_doc_start_end,
           'threshold_sents': threshold_sent_idxes}
    return res