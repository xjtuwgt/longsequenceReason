import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import *
distractor_wiki_path = '../data/hotpotqa/distractor_qa'
gold_wiki_path = '../data/hotpotqa/'
abs_distractor_wiki_path = os.path.abspath(distractor_wiki_path)
from pandas import DataFrame
from time import time

def Gold_Hotpot_Train_Dev_Data_Collection(data: DataFrame):
    def pos_context_extraction(row):
        supporting_facts, contexts = row['supporting_facts'], row['context']
        positive_titles = set([x[0] for x in supporting_facts])
        pos_context = []
        for ctx_idx, ctx in enumerate(contexts):  ## Original ctx index, record the original index order
            title, text = ctx[0], ctx[1]
            if title in positive_titles:
                pos_context.append(ctx)
        return pos_context
    data['context'] = data.swifter.apply(lambda row: pd.Series(pos_context_extraction(row)), axis=1)
    return data
#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def gold_doc_hotpotqa_extraction_example():
    start_time = time()
    dev_data, _ = HOTPOT_DevData_Distractor()
    print('*' * 75)
    gold_dev_data = Gold_Hotpot_Train_Dev_Data_Collection(data=dev_data)
    print('Get {} dev-test records'.format(gold_dev_data.shape[0]))
    gold_dev_data.to_json(os.path.join(gold_wiki_path, 'gold_hotpot_dev_distractor_v1.json'))
    print('Runtime = {:.4f} seconds'.format(time() - start_time))
    print('*' * 75)


if __name__ == '__main__':
    gold_doc_hotpotqa_extraction_example()
    # x = '100 bc  ad 400'
    # y = remove_multi_spaces(x)
    # print(len(x))
    # print(len(y))
    # hotpot_data_analysis()
    # num_of_support_sents()
    # answer_type()
    # data_consistent_checker()
    # data_statistic()
    print()