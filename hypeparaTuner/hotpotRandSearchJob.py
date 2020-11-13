import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import shutil
from hypeparaTuner.randSearch import RandomSearchJob

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def HypeParameterSpace():
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [4e-5, 5e-5, 1e-5]}
    adam_weight_decay = {'name': 'adam_weight_decay', 'type': 'fixed', 'value': 1e-6}
    feat_drop = {'name': 'fea_drop', 'type': 'choice', 'values': [0.1]}
    att_drop = {'name': 'att_drop', 'type': 'choice', 'values': [0.1]} #0.1
    project_dim = {'name': 'project_dim', 'type': 'choice', 'values': [256]}
    batch_size = {'name': 'batch_size', 'type': 'fixed', 'value': 8}
    max_doc_num = {'name': 'max_doc_num', 'type': 'fixed', 'value': 10}
    sent_threshold = {'name': 'sent_threshold', 'type': 'choice', 'values': [0.925, 0.95]}
    ir_name = {'name': 'score_model_name', 'type': 'choice', 'values': ['MLP']}
    task_name = {'name': 'task_name', 'type': 'choice', 'values': ['doc_sent', 'doc_sent_ans']} ## doc, doc_sent, doc_sent_ans
    trip_score_name = {'name': 'hop_model_name', 'type': 'fixed', 'value': 'DotProduct'}
    mask_type = {'name': 'mask_name', 'type': 'choice', 'values': ['query_doc_sent']} #'query', 'query_doc', 'query_doc_sent'
    frozen_layer_num = {'name': 'frozen_layer', 'type': 'choice', 'values': [2]} #1, 2
    span_weight = {'name': 'span_weight', 'type': 'choice', 'values': [0.1, 0.2, 0.5]}
    pair_score_weight = {'name': 'pair_score_weight', 'type': 'choice', 'values': [1.0]} #0.1, 0.2, 0.5, 1.0
    train_data_filtered = {'name': 'train_data', 'type': 'choice', 'values': [0]} # 0, 1, 2
    train_data_shuffler = {'name': 'train_shuffle', 'type': 'choice', 'values': [0]} # 0, 1
    with_graph = {'name': 'with_graph', 'type': 'choice', 'values': [0]} # 0, 1
    with_graph_training = {'name': 'with_graph_training', 'type': 'choice', 'values': [0]}# 0, 1
    epochs = {'name': 'epoch', 'type': 'fixed', 'value': 6}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, adam_weight_decay, att_drop, feat_drop, project_dim, trip_score_name, with_graph, with_graph_training,
                      batch_size, max_doc_num, epochs, sent_threshold, ir_name, mask_type, span_weight, pair_score_weight,
                    task_name,
                    frozen_layer_num, train_data_filtered, train_data_shuffler]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num):
    bash_save_path = '../hotpot_jobs/'
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    search_space = HypeParameterSpace()
    random_search_job =RandomSearchJob(search_space=search_space)
    for i in range(task_num):
        task_id, parameter_id = random_search_job.single_task_trial(i+200)
        with open(bash_save_path + 'run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash qarun.sh ' + parameter_id
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(task_num))

if __name__ == '__main__':
    generate_random_search_bash(task_num=6)