import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import save_check_point
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from time import time
import torch
from torch import Tensor as T
from pandas import DataFrame
from torch.utils.data import DataLoader
from modelEvaluation.hotpotEvaluationUtils import convert2leadBoard
from goldMultihopQA.goldHotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset
from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder, get_hotpotqa_longformer_tokenizer
from reasonModel.QAModel import LongformerHotPotQAModel
from modelEvaluation.hotpotEvaluationUtils import sp_score, answer_type_prediction, answer_span_prediction
from modelTrain.modelTrainUtils import read_train_dev_data_frame, log_metrics
########################################################################################################################
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
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels)
    logging.info('Constructing reasonModel completes in {:.4f}'.format(time() - start_time))
    return model

def training_warm_up(model, optimizer, train_dataloader, device, dev_dataloader, tokenizer, args):
    warm_up_steps = args.warm_up_steps
    start_time = time()
    step = 0
    training_logs = []
    logging.info('Starting warm up...')
    logging.info('*' * 75)
    ####################################################################################################################
    model.train()
    model.zero_grad()
    all_step_num = len(train_dataloader)
    ####################################################################################################################
    for batch_idx, train_sample in enumerate(train_dataloader):
        log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
        step = step + 1
        training_logs.append(log)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Training average', step, metrics)
            logging.info('Training in {}/{} ({}, {}) steps takes {:.4f} seconds'.format(step, all_step_num, 'warm_up', batch_idx + 1,
                                                                                     time() - start_time))
            training_logs = []
        if step >= warm_up_steps:
            logging.info('Warm up completed in {:.4f} seconds'.format(time() - start_time))
            logging.info('*' * 75)
            break
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, tokenizer=tokenizer, device=device, args=args)
    logging.info('*' * 75)
    logging.info('Answer type prediction accuracy: {}'.format(metric_dict['answer_type_acc']))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metrics'):
            logging.info('{} prediction'.format(key))
            log_metrics('Valid', 'warm up', value)
        logging.info('*' * 75)

def train_all_steps(model, optimizer, train_dataloader, dev_dataloader, device, tokenizer, args):
    assert args.save_checkpoint_steps % args.valid_steps == 0
    warm_up_steps = args.warm_up_steps
    if warm_up_steps > 0:
        training_warm_up(model=model, optimizer=optimizer, device=device, train_dataloader=train_dataloader, tokenizer=tokenizer, dev_dataloader=dev_dataloader, args=args)
        logging.info('*' * 75)
        current_learning_rate = optimizer.param_groups[-1]['lr']
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
    all_step_num = len(train_dataloader)
    for epoch_id in range(1, args.epoch + 1):
        for batch_idx, train_sample in enumerate(train_dataloader):
            log = train_single_step(model=model, optimizer=optimizer, train_sample=train_sample, args=args)
            # ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            scheduler.step()
            # ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            step = step + 1
            training_logs.append(log)
            torch.cuda.empty_cache()
            # ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.save_checkpoint_steps == 0:
                save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss, eval_metric=eval_metric, args=args)
                logging.info('Saving the mode in {}'.format(save_path))
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}/{}) steps takes {:.4f} seconds'.format(step, epoch_id, batch_idx + 1, all_step_num, time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Data set...')
                metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, device=device, args=args, tokenizer=tokenizer)
                logging.info('*' * 75)
                answer_type_acc = metric_dict['answer_type_acc']
                eval_metric = answer_type_acc
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                sent_pred_f1 = metric_dict['supp_sent_metrics']['sp_f1']
                logging.info('*' * 75)
                for key, value in metric_dict.items():
                    if key.endswith('metrics'):
                        logging.info('{} prediction'.format(key))
                        log_metrics('Valid', step, value)
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
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
    A single train step. Apply back-propagation and return the loss
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
    yn_loss, span_loss, supp_sent_loss = loss_output['answer_type_loss'], loss_output['span_loss'], loss_output['sent_loss']
    loss = supp_sent_loss + yn_loss + span_loss * args.span_weight
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'al_loss': loss.mean().item(),
        'an_loss': span_loss.mean().item(),
        'ss_loss': supp_sent_loss.mean().item(),
        'yn_loss': yn_loss.mean().item()
    }
    return log
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def test_all_steps(model, device, test_data_loader, tokenizer, args):
    '''
            Evaluate the reasonModel on test or valid datasets
    '''
    model.eval()
    ###########################################################
    start_time = time()
    test_dataset = test_data_loader
    sent_logs = []
    step = 0
    N = 0
    total_steps = len(test_dataset)
    support_sent_doc_sent_pair_results = []
    answer_type_pred_results = []
    span_pred_results = []
    encode_id_results = []
    # **********************************************************
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
            N = N + sample['sent_lens'].shape[0]
            eval_res = metric_computation(output_scores=output, sample=sample, args=args)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            correct_yn, yn_predicted_labels = eval_res['answer_type']
            correct_answer_num += correct_yn
            answer_type_pred_results += yn_predicted_labels
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_metric_logs, sent_pair_pred = eval_res['supp_sent']
            sent_logs += sent_metric_logs
            support_sent_doc_sent_pair_results += sent_pair_pred
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            span_predicted_i = eval_res['answer_span']
            span_pred_results += span_predicted_i
            # ******************************************
            encode_ids = eval_res['encode_ids']
            encode_id_results += encode_ids
            # ******************************************
            step += 1
            if step % args.test_log_steps == 0:
                logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    sent_metrics = {}
    for metric in sent_logs[0].keys():
        sent_metrics[metric] = sum([log[metric] for log in sent_logs]) / len(sent_logs)
    ##=================================================
    answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
    result_dict = {'aty_pred': answer_type_pred_results,
                   'ans_span': span_pred_results,
                   'ss_ds_pair': support_sent_doc_sent_pair_results,
                   'encode_ids': encode_id_results} ## for detailed results checking
    res_data_frame = DataFrame(result_dict)
    print(res_data_frame.shape)
    leadboard_metric, _ = convert2leadBoard(data=res_data_frame, tokenizer=tokenizer, gold=True)
    return {'supp_sent_metrics': sent_metrics, 'answer_type_acc': answer_type_accuracy, 'lead_board_metrics': leadboard_metric}

def metric_computation(output_scores: dict, sample: dict, args):
    # =========Answer type prediction==========================
    yn_scores = output_scores['answer_type_score']
    yn_true_labels = sample['yes_no']
    if len(yn_true_labels.shape) > 1:
        yn_true_labels = yn_true_labels.squeeze(dim=-1)
    correct_yn, yn_predicted_labels = answer_type_prediction(type_scores=yn_scores, true_labels=yn_true_labels)
    # =========Answer span prediction==========================
    start_logits, end_logits = output_scores['span_score']
    sent_start_position, sent_end_position, sent_lens = sample['sent_start'], sample['sent_end'], sample['sent_lens']
    predicted_span_pair = answer_span_prediction(start_scores=start_logits, end_scores=end_logits,
                                                 sent_start_positions=sent_start_position, sent_end_positions=sent_end_position, sent_mask=sent_lens)
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    supp_sent_scores = output_scores['sent_score']
    sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
    sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
    sent_fact_doc_idx, sent_fact_sent_idx = sample['s2d_map'], sample['sInd_map']
    sent_metric_logs, _, supp_sent_pred_pair = support_sent_evaluation(scores=supp_sent_scores, labels=sent_label, mask=sent_mask, pred_num=2,
                                                               threshold=args.sent_threshold, doc_fact=sent_fact_doc_idx, sent_fact=sent_fact_sent_idx)
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    encode_ids = sample['ctx_encode'].detach().tolist()
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    return {'answer_type': (correct_yn, yn_predicted_labels),
            'answer_span': predicted_span_pair,
            'supp_sent': (sent_metric_logs, supp_sent_pred_pair),
            'encode_ids': encode_ids}

def support_sent_evaluation(scores: T, labels: T, mask: T, doc_fact: T, sent_fact: T, pred_num=2, threshold=0.8):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    predicted_labels = []
    predicted_label_pairs = []
    for idx in range(batch_size):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_fact_i = doc_fact[idx].detach().tolist()
        sent_fact_i = sent_fact[idx].detach().tolist()
        doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        for i in range(pred_num, sample_size):
            if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
                pred_labels_i.append(pred_idxes_i[i])
        labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        predicted_labels.append(pred_labels_i)
        pred_labels_pair_i = [doc_sent_pair_i[_] for _ in pred_labels_i]
        predicted_label_pairs.append(pred_labels_pair_i)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
        logs.append({
            'sp_em': em_i,
            'sp_f1': f1_i,
            'sp_prec': prec_i,
            'sp_recall': recall_i
        })
    return logs, predicted_labels, predicted_label_pairs
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++