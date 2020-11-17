import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import logging
from time import time
import torch
from pandas import DataFrame
from torch import Tensor as T
from modelEvaluation.hotpotEvaluationUtils import answer_type_prediction, answer_span_prediction
from modelEvaluation.hotpotEvaluationUtils import sp_score
from modelEvaluation.hotpotEvaluationUtils import convert2leadBoard
from transformers import LongformerTokenizer
##################################
MASK_VALUE = -1e9
##################################
def multi_task_decoder(model, test_data_loader, tokenizer, device, args):
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
    support_doc_pred_results = []
    support_sent_pred_results, support_sent_doc_sent_pair_results = [], []
    answer_type_pred_results = []
    span_pred_results = []
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
            eval_res = hotpot_prediction(output_scores=output, sample=sample, args=args)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            correct_yn, yn_predicted_labels = eval_res['answer_type']
            correct_answer_num += correct_yn
            answer_type_pred_results += yn_predicted_labels
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            span_predicted_i = eval_res['answer_span']
            span_pred_results += span_predicted_i
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            encode_ids = eval_res['encode_ids']
            encode_id_results += encode_ids
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            doc_metric_logs, doc_pred_res = eval_res['supp_doc']
            doc_logs += doc_metric_logs
            doc_predicted_labels = doc_pred_res
            support_doc_pred_results += doc_predicted_labels
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_metric_logs, sent_pred_res = eval_res['supp_sent']
            sent_logs += sent_metric_logs
            sent_predicted_labels, doc_sent_fact_pair = sent_pred_res
            support_sent_pred_results += sent_predicted_labels
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
                   'ans_span': span_pred_results,
                   'ss_ds_pair': support_sent_doc_sent_pair_results,
                   'encode_ids': encode_id_results} ## for detailed results checking
    res_data_frame = DataFrame(result_dict)
    ##################################################
    leadboard_metric, res_data_frame = convert2leadBoard(data=res_data_frame, tokenizer=tokenizer)
    ##=================================================
    return {'supp_doc_metrics': doc_metrics, 'supp_sent_metrics': sent_metrics,
            'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}

def hotpot_prediction(output_scores: dict, sample: dict, args):
    # =========Answer type prediction==========================
    yn_scores = output_scores['yn_score']
    yn_true_labels = sample['yes_no']
    if len(yn_true_labels.shape) > 1:
        yn_true_labels = yn_true_labels.squeeze(dim=-1)
    correct_num, type_predicted_labels = answer_type_prediction(type_scores=yn_scores, true_labels=yn_true_labels) ## yes, no, span
    # =========Answer span prediction==========================
    start_logits, end_logits = output_scores['span_score']
    sent_start_position, sent_end_position, sent_lens = sample['sent_start'], sample['sent_end'], sample['sent_lens']
    predicted_span_pair = answer_span_prediction(start_scores=start_logits, end_scores=end_logits,
                                                 sent_start_positions=sent_start_position, sent_end_positions=sent_end_position, sent_mask=sent_lens)
    # =========Answer span prediction==========================
    doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
    doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
    supp_doc_scores, _ = output_scores['doc_score']
    doc_metric_logs, doc_pred_res = support_doc_evaluation(scores=supp_doc_scores, labels=doc_label, mask=doc_mask,
                                                           pred_num=2)
    # +++++++++ supp doc prediction +++++++++++++++++++++++++++
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    supp_sent_scores = output_scores['sent_score']
    sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
    sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
    sent_fact_doc_idx, sent_fact_sent_idx = sample['s2d_map'], sample['sInd_map']
    sent_metric_logs, _, sent_pred_res = support_sent_evaluation(scores=supp_sent_scores, labels=sent_label,
                                                                 mask=sent_mask, pred_num=2,
                                                                 threshold=args.sent_threshold,
                                                                 doc_fact=sent_fact_doc_idx,
                                                                 sent_fact=sent_fact_sent_idx)
    # +++++++++ supp sent prediction +++++++++++++++++++++++++++
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    encode_ids = sample['ctx_encode'].detach().tolist()
    # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
    return {'answer_type': (correct_num, type_predicted_labels),
            'answer_span': predicted_span_pair,
            'supp_doc': (doc_metric_logs, doc_pred_res),
            'supp_sent': (sent_metric_logs, sent_pred_res),
            'encode_ids': encode_ids}

def support_doc_evaluation(scores: T, labels: T, mask: T, pred_num=2):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    logs = []
    predicted_labels = []
    score_list = []
    for idx in range(batch_size):
        score_list.append(masked_scores[idx].detach().tolist())
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        predicted_labels.append(pred_labels_i)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
        logs.append({
            'sp_em': em_i,
            'sp_f1': f1_i,
            'sp_prec': prec_i,
            'sp_recall':recall_i
        })
    return logs, predicted_labels
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
        pred_labels_pair_i = [doc_sent_pair_i[_] for _ in predicted_labels]
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