import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from time import time
import pandas as pd
from pandas import DataFrame
from multihopUtils.hotpotqaIOUtils import HOTPOT_DevData_Distractor
from multihopUtils.hotpotqaIOUtils import loadWikiData as load_data_frame
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE, get_hotpotqa_longformer_tokenizer
from transformers import LongformerTokenizer
from modelEvaluation.hotpot_evaluate_v1 import json_eval
import swifter
span_length_limit = 20

def load_data_frame_align_with_dev(file_path, json_fileName):
    start_time = time()
    data_frame = load_data_frame(PATH=file_path, json_fileName=json_fileName)
    orig_dev_data, _ = HOTPOT_DevData_Distractor()
    assert orig_dev_data.shape[0] == data_frame.shape[0]
    data_frame = add_row_idx(data_frame)
    orig_dev_data = add_row_idx(orig_dev_data)
    data = data_frame.merge(orig_dev_data, left_on='row_idx', right_on='row_idx')
    print('Merging {} in {:.4f} seconds'.format(data.shape, time() - start_time))
    return data

def get_all_json_files(file_path: str, extension: str = '.json'):
    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith(extension)]
    return file_names

def add_row_idx(data: DataFrame):
    data['row_idx'] = range(0, len(data))
    return data
def print_metrics(name: str, metrics: dict):
    for key, value in metrics.items():
        print('{}: {}: {}'.format(name, key, value))
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def performance_collection(folder_name):
    print('Loading tokenizer')
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    json_file_names = get_all_json_files(file_path=folder_name)
    json_file_names = [x for x in json_file_names if x != 'config.json']
    print('{} json files have been found'.format(len(json_file_names)))
    max_sp_sent_f1 = 0
    max_metric_res = None
    max_json_file_name = None
    for idx, json_file_name in enumerate(json_file_names):
        if json_file_name != 'config.json':
            data_frame_i = load_data_frame_align_with_dev(file_path=folder_name, json_fileName=json_file_name)
            metrics_i = convert2leadBoard(data=data_frame_i, tokenizer=tokenizer)
            if max_sp_sent_f1 < metrics_i['sp_f1']:
                max_sp_sent_f1 = metrics_i['sp_f1']
                max_metric_res = metrics_i
                max_json_file_name = json_file_name
            print_metrics(name=json_file_name, metrics=metrics_i)
            print('*'*75)
    print('+' * 75)
    print_metrics(name=max_json_file_name, metrics=max_metric_res)
    print('+' * 75)

def convert2leadBoard(data: DataFrame, tokenizer: LongformerTokenizer):
    def process_row(row):
        answer_type_prediction = row['aty_pred']
        support_doc_prediction = row['sd_pred']
        support_sent_prediction = row['ss_pred']
        ss_ds_pair = row['ss_ds_pair']
        span_prediction = row['ans_span']
        encode_ids = row['encode_ids']
        context_docs = row['context']
        if answer_type_prediction == 0:
            span_start, span_end = span_prediction[0], span_prediction[1]
            if span_end - span_start > 20:
                span_end = span_start + 20
            answer_encode_ids = encode_ids[span_start:(span_end+1)]
            answer_prediction = tokenizer.decode(answer_encode_ids, skip_special_tokens=True)
        elif answer_type_prediction == 1:
            answer_prediction = 'yes'
        else:
            answer_prediction = 'no'

        supp_doc_titles = [context_docs[idx][0] for idx in support_doc_prediction]
        supp_sent_prediction_pair = [ss_ds_pair[idx] for idx in support_sent_prediction]
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
    return metrics


def convert2leadboard_hierartical(data: DataFrame, tokenizer: LongformerTokenizer):

    def process_row(row):
        answer_type_prediction = row['aty_pred']
        topk_support_doc_prediction = row['topk_sd_pred']
        thresh_support_doc_prediction = row['thresh_sd_pred']

        topk_support_sent_prediction = row['topk_ss_pred']
        thresh_support_sent_prediction = row['thresh_ss_pred']
        topk_ss_ds_pair = row['topk_ds_pair']
        thresh_ss_ds_pair = row['thresh_ds_pair']
        topk_ans_span = row['topk_ans_span']
        thresh_ans_span = row['thresh_ans_span']
        encode_ids = row['encode_ids']
        context_docs = row['context']
        if answer_type_prediction == 0:
            topk_span_start, topk_span_end = topk_ans_span[0], topk_ans_span[1]
            if topk_span_end > topk_span_start + span_length_limit:
                topk_span_end = topk_span_start + span_length_limit
            topk_answer_encode_ids = encode_ids[topk_span_start:(topk_span_end + 1)]
            topk_answer_prediction = tokenizer.decode(topk_answer_encode_ids, skip_special_tokens=True)

            thresh_span_start, thresh_span_end = thresh_ans_span[0], thresh_ans_span[1]
            if thresh_span_end > thresh_span_start + span_length_limit:
                thresh_span_end = thresh_span_start + span_length_limit
            thresh_answer_encode_ids = encode_ids[thresh_span_start:(thresh_span_end + 1)]
            thresh_answer_prediction = tokenizer.decode(thresh_answer_encode_ids, skip_special_tokens=True)

        elif answer_type_prediction == 1:
            topk_answer_prediction = 'yes'
            thresh_answer_prediction = 'yes'
        else:
            topk_answer_prediction = 'no'
            thresh_answer_prediction = 'no'

        topk_supp_doc_titles = [context_docs[idx][0] for idx in topk_support_doc_prediction]
        topk_supp_sent_prediction_pair = [topk_ss_ds_pair[idx] for idx in topk_support_sent_prediction]
        topk_supp_title_sent_id = [(context_docs[x[0]][0], x[1]) for x in topk_supp_sent_prediction_pair]

        thresh_supp_doc_titles = [context_docs[idx][0] for idx in thresh_support_doc_prediction]
        thresh_supp_sent_prediction_pair = [thresh_ss_ds_pair[idx] for idx in thresh_support_sent_prediction]
        thresh_supp_title_sent_id = [(context_docs[x[0]][0], x[1]) for x in thresh_supp_sent_prediction_pair]
        return topk_answer_prediction, topk_supp_doc_titles, topk_supp_title_sent_id, thresh_answer_prediction, thresh_supp_doc_titles, thresh_supp_title_sent_id

    pred_names = ['topk_answer', 'topk_sp_doc', 'topk_sp', 'thresh_answer', 'thresh_sp_doc', 'thresh_sp']
    data[pred_names] = data.swifter.apply(lambda row: pd.Series(process_row(row)), axis=1)

    golden_data, _ = HOTPOT_DevData_Distractor()
    golden_data_dict = golden_data.to_dict(orient='records')
    ##++++++++++++++++++++++++++++++++
    topk_res_names = ['_id', 'topk_answer', 'topk_sp_doc', 'topk_sp']
    predicted_data = data[topk_res_names]
    id_list = predicted_data['_id'].tolist()
    answer_list = predicted_data['topk_answer'].tolist()
    sp_list = predicted_data['topk_sp'].tolist()
    answer_id_dict = dict(zip(id_list, answer_list))
    sp_id_dict = dict(zip(id_list, sp_list))
    predicted_data_dict = {'answer': answer_id_dict, 'sp': sp_id_dict}
    topk_metrics = json_eval(prediction=predicted_data_dict, gold=golden_data_dict)
    ##++++++++++++++++++++++++++++++++
    topk_res_names = ['_id', 'thresh_answer', 'thresh_sp_doc', 'thresh_sp']
    predicted_data = data[topk_res_names]
    id_list = predicted_data['_id'].tolist()
    answer_list = predicted_data['thresh_answer'].tolist()
    sp_list = predicted_data['thresh_sp'].tolist()
    answer_id_dict = dict(zip(id_list, answer_list))
    sp_id_dict = dict(zip(id_list, sp_list))
    predicted_data_dict = {'answer': answer_id_dict, 'sp': sp_id_dict}
    thresh_metrics = json_eval(prediction=predicted_data_dict, gold=golden_data_dict)
    return topk_metrics, thresh_metrics

###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
   result_folder_name = '../model/'
   performance_collection(folder_name=result_folder_name)
   print()