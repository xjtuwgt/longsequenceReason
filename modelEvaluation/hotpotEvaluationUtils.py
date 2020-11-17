import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import pandas as pd
from pandas import DataFrame
from multihopUtils.hotpotqaIOUtils import HOTPOT_DevData_Distractor
from transformers import LongformerTokenizer
from modelEvaluation.hotpot_evaluate_v1 import json_eval
from torch import Tensor as T
import torch
import torch.nn.functional as F
import swifter
MAX_ANSWER_DECODE_LEN = 50

########################################################################################################################
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
########################################################################################################################

def answer_type_prediction(type_scores: T, true_labels: T):
    type_predicted_labels = torch.argmax(type_scores, dim=-1)
    correct_num = (type_predicted_labels == true_labels).sum().data.item()
    type_predicted_labels = type_predicted_labels.detach().tolist()
    ans_type_map = {0: 'span', 1: 'yes', 2: 'no'}
    type_predicted_labels = [ans_type_map[_] for _ in type_predicted_labels]
    return correct_num, type_predicted_labels

def answer_span_prediction(start_scores: T, end_scores: T, sent_start_positions: T, sent_end_positions: T, sent_mask: T):
    batch_size, seq_len = start_scores.shape[0], start_scores.shape[0]
    start_prob = torch.sigmoid(start_scores)
    end_prob = torch.sigmoid(end_scores)
    sent_number = sent_start_positions.shape[0]
    if len(sent_start_positions.shape) > 1:
        sent_start_positions = sent_start_positions.unsqueeze(dim=-1)
    if len(sent_end_positions.shape) > 1:
        sent_end_positions = sent_end_positions.unsqueeze(dim=-1)
    answer_span_pairs = []
    for batch_idx in range(batch_size):
        max_score_i = 0
        max_pair_idx = None
        for sent_idx in range(sent_number):
            if sent_mask[batch_idx][sent_idx] > 0:
                sent_start_i, sent_end_i = sent_start_positions[batch_idx][sent_idx], sent_end_positions[batch_idx][sent_idx]
                sent_start_score_i = start_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
                sent_end_score_i = end_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
                max_sent_core_i, start_idx, end_idx = answer_span_in_sentence(start_scores=sent_start_score_i, end_scores=sent_end_score_i)
                start_idx = start_idx + sent_start_i
                end_idx = end_idx + sent_end_i
                if max_score_i < max_sent_core_i:
                    max_pair_idx = (start_idx, end_idx)
        answer_span_pairs.append(max_pair_idx)
    return answer_span_pairs

def answer_span_in_sentence(start_scores: T, end_scores: T, max_ans_decode_len: int = MAX_ANSWER_DECODE_LEN):
    sent_len = start_scores.shape[0]
    score_matrix = torch.matmul(start_scores.view(1,-1).t(), end_scores.view(1,-1))
    score_matrix = torch.triu(score_matrix)
    if max_ans_decode_len < sent_len:
        trip_len = sent_len - max_ans_decode_len
        mask_upper_tri = torch.triu(torch.ones((trip_len, trip_len))).to(start_scores.device)
        mask_upper_tri = F.pad(mask_upper_tri, [max_ans_decode_len, 0, 0, max_ans_decode_len])
        score_matrix = score_matrix.masked_fill(mask_upper_tri==1, 0)
    max_idx = torch.argmax(score_matrix)
    start_idx, end_idx = max_idx // sent_len, max_idx % sent_len
    start_idx, end_idx = start_idx.data.item(), end_idx.data.item()
    score = score_matrix[start_idx][end_idx]
    return score, start_idx, end_idx

def add_id_context(data: DataFrame):
    golden_data, _ = HOTPOT_DevData_Distractor()
    data[['_id', 'context']] = golden_data[['_id', 'context']]
    return data

def convert2leadBoard(data: DataFrame, tokenizer: LongformerTokenizer):
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    data = add_id_context(data=data)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_row(row):
        answer_type_prediction = row['aty_pred']
        support_doc_prediction = row['sd_pred']
        support_sent_prediction = row['ss_pred']
        ss_ds_pair = row['ss_ds_pair']
        assert len(support_sent_prediction) == len(ss_ds_pair)
        supp_sent_prediction_pair = ss_ds_pair
        span_prediction = row['ans_span']
        encode_ids = row['encode_ids']
        context_docs = row['context']
        if answer_type_prediction == 'span':
            span_start, span_end = span_prediction[0], span_prediction[1]
            answer_encode_ids = encode_ids[span_start:(span_end+1)]
            answer_prediction = tokenizer.decode(answer_encode_ids, skip_special_tokens=True)
            answer_prediction = answer_prediction.strip()
        else:
            answer_prediction = answer_type_prediction

        supp_doc_titles = [context_docs[idx][0] for idx in support_doc_prediction]
        supp_title_sent_id = [(context_docs[x[0]][0], x[1]) for x in supp_sent_prediction_pair]
        return answer_prediction, supp_doc_titles, supp_title_sent_id

    pred_names = ['answer', 'sp_doc', 'sp']
    data[pred_names] = data.swifter.apply(lambda row: pd.Series(process_row(row)), axis=1)
    res_names = ['_id', 'answer', 'sp_doc', 'sp']

    predicted_data = data[res_names]
    id_list = predicted_data['_id'].tolist()
    answer_list = predicted_data['answer'].tolist()
    sp_list = predicted_data['sp'].tolist()
    answer_id_dict = dict(zip(id_list, answer_list))
    sp_id_dict = dict(zip(id_list, sp_list))
    predicted_data_dict = {'answer': answer_id_dict, 'sp': sp_id_dict}
    golden_data, _ = HOTPOT_DevData_Distractor()
    golden_data_dict = golden_data.to_dict(orient='records')
    metrics = json_eval(prediction=predicted_data_dict, gold=golden_data_dict)
    res_data_frame = pd.DataFrame.from_dict(predicted_data_dict)
    return metrics, res_data_frame
