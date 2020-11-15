import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import load_model
import logging
import argparse
import os
from time import time
import torch
import pandas as pd
import json
from torch import Tensor as T
from modelTrain.QATrainFunction import get_date_time, read_train_dev_data_frame
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE, get_hotpotqa_longformer_tokenizer

from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from torch.utils.data import DataLoader
from multihopQA.hotpotQAdataloader import HotpotDevDataset, HotpotTestDataset


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hotpot_test_prediction(model, test_data_loader, args):
    model.eval()
    ###########################################################
    start_time = time()
    step = 0
    N = 0
    total_steps = len(test_data_loader)
    # **********************************************************
    answer_type_predicted = []
    answer_span_predicted = []
    supp_sent_predicted = []
    supp_doc_predicted = []
    # **********************************************************
    with torch.no_grad():
        for test_sample in test_data_loader:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    sample[key] = value.cuda()
            else:
                sample = test_sample
            output = model(sample)
            N = N + sample['ctx_encode'].shape[0]
            # ++++++++++++++++++
            answer_type_res = output['yn_score']
            if len(answer_type_res.shape) > 1:
                answer_type_res = answer_type_res.squeeze(dim=-1)
            answer_types = torch.argmax(answer_type_res, dim=-1)
            answer_type_predicted += answer_types.detach().tolist()
            # +++++++++++++++++++
            start_logits, end_logits = output['span_score']
            predicted_span_start = torch.argmax(start_logits, dim=-1)
            predicted_span_end = torch.argmax(end_logits, dim=-1)
            predicted_span_start = predicted_span_start.detach().tolist()
            predicted_span_end = predicted_span_end.detach().tolist()
            predicted_span_pair = list(zip(predicted_span_start, predicted_span_end))
            answer_span_predicted += predicted_span_pair
            # ++++++++++++++++++
            supp_doc_res = output['doc_score']
            doc_lens = sample['doc_lens']
            doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
            supp_doc_pred_i = supp_doc_prediction(scores=supp_doc_res, mask=doc_mask, pred_num=2)
            supp_doc_predicted += supp_doc_pred_i
            # ++++++++++++++++++
            supp_sent_res = output['sent_score']
            sent_lens = sample['sent_lens']
            sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
            sent_fact_doc_idx, sent_fact_sent_idx = sample['fact_doc'], sample['fact_sent']
            supp_sent_pred_i = supp_sent_prediction(scores=supp_sent_res, mask=sent_mask, doc_fact=sent_fact_doc_idx,
                                                    sent_fact=sent_fact_sent_idx, pred_num=2, threshold=args.sent_threshold)
            supp_sent_predicted += supp_sent_pred_i
            # +++++++++++++++++++
            step += 1
            if step % args.test_log_steps == 0:
                logging.info(
                    'Testing the reasonModel... {}/{} in {:.4f} seconds'.format(step, total_steps, time() - start_time))
    ##=================================================
    logging.info('Testing complete...')
    logging.info('Loading tokenizer')
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    logging.info('Loading preprocessed data...')
    data = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.test_data_name)
    data['answer_prediction'] = answer_type_predicted
    data['answer_span_prediction'] = answer_span_predicted
    data['supp_doc_prediction'] = supp_doc_predicted
    data['supp_sent_prediction'] = supp_sent_predicted

    def row_process(row):
        answer_prediction = row['answer_prediction']
        answer_span_predicted = row['answer_span_prediction']
        span_start, span_end = answer_span_predicted
        encode_ids = row['ctx_encode']
        if answer_prediction > 0:
            predicted_answer = 'yes' if answer_prediction == 1 else 'no'
        else:
            predicted_answer = tokenizer.decode(encode_ids[span_start:(span_end+1)], skip_special_tokens=True)

        ctx_contents = row['context']
        supp_doc_prediction = row['supp_doc_prediction']
        supp_doc_titles = [ctx_contents[idx][0] for idx in supp_doc_prediction]
        supp_sent_prediction = row['supp_sent_prediction']
        supp_sent_pairs = [(ctx_contents[pair_idx[0]][0], pair_idx[1])for pair_idx in supp_sent_prediction]
        return predicted_answer, supp_doc_titles, supp_sent_pairs

    res_names = ['answer', 'sp_doc', 'sp']
    data[res_names] = data.apply(lambda row: pd.Series(row_process(row)), axis=1)
    return data

def supp_doc_prediction(scores: T, mask: T, pred_num=2):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    supp_facts_predicted = []
    for idx in range(batch_size):
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        supp_facts_predicted.append(pred_labels_i)
    return supp_facts_predicted

def supp_sent_prediction(scores: T, mask: T, doc_fact: T, sent_fact: T, pred_num=2, threshold=0.9):
    batch_size, sample_size = scores.shape[0], scores.shape[1]
    scores = torch.sigmoid(scores)
    masked_scores = scores.masked_fill(mask == 0, -1)
    argsort = torch.argsort(masked_scores, dim=1, descending=True)
    supp_facts_predicted = []
    for idx in range(batch_size):
        pred_idxes_i = argsort[idx].tolist()
        pred_labels_i = pred_idxes_i[:pred_num]
        for i in range(pred_num, sample_size):
            if masked_scores[idx, pred_idxes_i[i]] >= threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
                pred_labels_i.append(pred_idxes_i[i])
        #################################################
        doc_fact_i = doc_fact[idx].detach().tolist()
        sent_fact_i = sent_fact[idx].detach().tolist()
        doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i))  ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
        #################################################
        doc_sent_idx_pair_i = []
        for pred_idx in pred_labels_i:
            doc_sent_idx_pair_i.append(doc_sent_pair_i[pred_idx])
        supp_facts_predicted.append(doc_sent_idx_pair_i)
    return supp_facts_predicted