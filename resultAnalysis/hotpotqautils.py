import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from modelEvaluation.hotpot_evaluation_collections import performance_collection

MODEL_PATH = '../model'
def get_all_folders(path: str):
    directory_contents = os.listdir(path)
    folder_list = []
    for item in directory_contents:
        if os.path.isdir(os.path.join(path, item)):
            folder_list.append(item)
    print('{} folders have been found'.format(len(folder_list)))
    print('*' * 75)
    return folder_list

def get_file_distribution(path: str):
    directory_contents = os.listdir(path)
    file_type_dict = dict()
    for item in directory_contents:
        if not os.path.isdir(os.path.join(path, item)):
            last_idx = item.rindex('.')
            # print(last_idx)
            file_type_name = item[last_idx:]
            if file_type_name not in file_type_dict:
                file_type_dict[file_type_name] = 1
            else:
                file_type_dict[file_type_name] = file_type_dict[file_type_name] + 1
    return file_type_dict

def get_all_json_files(file_path: str, extension: str = '.json'):
    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith(extension)]
    return file_names

def get_all_log_files(file_path: str, extension: str = '.log'):
    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith(extension)]
    return file_names

def result_folder_analysis(path: str):
    folder_name_list = get_all_folders(path=path)
    folder_with_json, folder_wo_json = [], []
    for folder_idx, folder_name in enumerate(folder_name_list):
        file_type_dict = get_file_distribution(path=os.path.join(path, folder_name))
        if '.json' in file_type_dict:
            folder_with_json.append((folder_name, file_type_dict))
            # print('{}-th folder with {} json files, folder name = {}'.format(folder_idx + 1,  file_type_dict['.json'], folder_name))
        else:
            folder_wo_json.append((folder_name, file_type_dict))
            # print('{}-th folder without json files, folder name = {}'.format(folder_idx + 1, folder_name))
    print('$' * 75)
    print('{} folders with json, and {} folders without json'.format(len(folder_with_json), len(folder_wo_json)))
    print('$' * 75)
    for folder_idx, folder_content in enumerate(folder_with_json):
        folder_name, file_type_dict = folder_content
        print('{}-th folder with {} json files, folder name = {}'.format(folder_idx + 1, file_type_dict['.json'],
                                                                         folder_name))
        print('-' * 75)
    print('*' * 75)
    # for folder_idx, folder_content in enumerate(folder_wo_json):
    #     folder_name, file_type_dict = folder_content
    #     print('{}-th folder without json files, folder name = {}'.format(folder_idx + 1, folder_name))
    #     print('-' * 75)
    return folder_with_json, folder_wo_json

def log_analysis_example(path: str):
    folder_with_json, _ = result_folder_analysis(path=path)
    max_supp_sent_f1_metric = 0
    max_folder_name = None
    max_doc, max_sent, max_ans_type = None, None, None
    for folder_idx, folder_content in enumerate(folder_with_json):
        folder_name, file_type_dict = folder_content
        if file_type_dict['.log'] == 1:
            log_file_names = get_all_log_files(file_path=os.path.join(path, folder_name))
            log_file_name = log_file_names[0]
            supp_sent_f1_metric, max_doc_pred, max_sent_pred, max_answer_type = log_analysis(os.path.join(path, folder_name, log_file_name))
            if max_supp_sent_f1_metric < supp_sent_f1_metric:
                max_supp_sent_f1_metric = supp_sent_f1_metric
                max_folder_name = folder_name
                max_doc, max_sent, max_ans_type = max_doc_pred, max_sent_pred, max_answer_type
        print('=' * 75)
    return max_folder_name, max_supp_sent_f1_metric, max_doc, max_sent, max_ans_type


def log_analysis(log_file_name: str):
    count = 0
    support_doc_metric = 'supp_doc_metrics'
    support_sent_metric = 'supp_sent_metrics'
    answer_type_metric = 'Answer type prediction accuracy'
    valid_metric = 'Valid'

    supp_doc_metric_list, supp_sent_metric_list, answer_type_list = [], [], []
    with open(log_file_name) as fp:
        while True:
            # read line
            line = fp.readline()
            count = count + 1
            if not line:
                break
            if support_doc_metric in line:
                line_num = 0
                doc_metrics = []
                while (line_num < 5) and (line is not None):
                    line = fp.readline()
                    if valid_metric in line:
                        tokens = line.split(': ')
                        assert len(tokens) == 2
                        metric = float(tokens[-1].strip())
                        doc_metrics.append(metric)
                    line_num = line_num + 1
                supp_doc_metric_list.append(doc_metrics)
            elif support_sent_metric in line:
                line_num = 0
                sent_metrics = []
                while (line_num < 5) and (line is not None):
                    line = fp.readline()
                    if valid_metric in line:
                        tokens = line.split(': ')
                        assert len(tokens) == 2
                        metric = float(tokens[-1].strip())
                        sent_metrics.append(metric)
                    line_num = line_num + 1
                supp_sent_metric_list.append(sent_metrics)
            elif answer_type_metric in line:
                tokens = line.split(': ')
                assert len(tokens) == 2
                metric = float(tokens[-1].strip())
                answer_type_list.append(metric)
            else:
                continue
    # print(len(supp_doc_metric_list), len(supp_sent_metric_list), len(answer_type_list), count)
    assert len(supp_sent_metric_list) == len(supp_sent_metric_list) and len(supp_sent_metric_list) == len(answer_type_list)
    max_supp_f1_idx = 0
    max_supp_f1 = 0
    max_doc_prediction = None
    max_sent_prediciton = None
    max_answer_type_prediction = None
    for idx, x in enumerate(supp_sent_metric_list):
        if x[1] > max_supp_f1:
            max_supp_f1 = x[1]
            max_supp_f1_idx = idx
            max_doc_prediction = supp_doc_metric_list[idx]
            max_sent_prediciton = supp_sent_metric_list[idx]
            max_answer_type_prediction = answer_type_list[idx]
    print('Support document prediction: {}'.format(supp_doc_metric_list[max_supp_f1_idx]))
    print('Support sentence prediction: {}'.format(supp_sent_metric_list[max_supp_f1_idx]))
    print('Answer type prediction: {}'.format(answer_type_list[max_supp_f1_idx]))
    print('Log file name: {}'.format(log_file_name))
    return max_supp_f1, max_doc_prediction, max_sent_prediciton, max_answer_type_prediction


def json_analysis_example(path: str):
    max_folder_name, max_supp_sent_f1, max_doc, max_sent, max_answer_type = log_analysis_example(path=MODEL_PATH)
    performance_collection(os.path.join(path, max_folder_name))
    return

def max_log_analysis_example(path: str):
    result_folder_analysis(path=path)
    # max_folder_name, max_supp_sent_f1, max_doc, max_sent, max_answer_type = log_analysis_example(path=path)
    # print('Best performance setting = {}\nwith metric = {}'.format(max_folder_name, max_supp_sent_f1))
    # print('Support document prediction: {}'.format(max_doc))
    # print('Support sentence prediction: {}'.format(max_sent))
    # print('Answer type prediction: {}'.format(max_answer_type))
    # performance_collection(os.path.join(path, max_folder_name))
    # performance_collection2(os.path.join(path, max_folder_name))

if __name__ == '__main__':
    # get_all_folders(path=MODEL_PATH)
    # result_folder_analysis(path=MODEL_PATH)
    max_log_analysis_example(path=MODEL_PATH)
    print()