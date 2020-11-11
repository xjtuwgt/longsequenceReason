import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import *
distractor_wiki_path = '../data/hotpotqa/distractor_qa'
abs_distractor_wiki_path = os.path.abspath(distractor_wiki_path)
from pandas import DataFrame
from time import time
import swifter
from multihopUtils.longformerQAUtils import LongformerQATensorizer
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
from multihopUtils.longformerQAUtils import get_hotpotqa_longformer_tokenizer
from transformers import LongformerTokenizer
import itertools
import operator
import numpy as np

def hotpot_data_analysis():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    col_list = []
    for col in data_frame.columns:
        col_list.append(col)
    print(col_list)
    max_len = 4096
    padding_num = 0.0
    max_doc_len = 0
    for idx, row in data_frame.iterrows():
        p_ctx_len, n_ctx_len = row['p_ctx_lens'], row['n_ctx_lens']
        q_len = row['ques_len']
        ctx_lens = p_ctx_len + n_ctx_len
        ctx_len_sum = sum(ctx_lens) + q_len
        if max_doc_len < ctx_len_sum:
            max_doc_len = ctx_len_sum
        if ctx_len_sum < max_len:
            padding_num = padding_num + 1
        else:
            print(ctx_len_sum)
    print(padding_num/data_frame.shape[0], padding_num, data_frame.shape[0])
    print(max_doc_len)

def num_of_support_sents():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    supp_sent_num_list = []
    supp_sent_array = np.zeros(100)
    for idx, row in data_frame.iterrows():
        supp_facts = row['supp_facts_filtered']
        supp_sent_num_list.append(len(supp_facts))
        supp_sent_array[len(supp_facts)] = supp_sent_array[len(supp_facts)] + 1

    percitile_array = [95, 97.5, 99.5, 99.75, 99.9, 99.99]
    supp_sent_num = np.array(supp_sent_num_list)
    min_values = ('min', np.min(supp_sent_num))
    max_values = ('max', np.max(supp_sent_num))
    pc_list = []
    pc_list.append(min_values)
    for pc in percitile_array:
        pc_list.append((pc, np.percentile(supp_sent_num, pc)))
    pc_list.append(max_values)
    print(pc_list)
    for i in range(100):
        if supp_sent_array[i] > 0:
            print('{}\t{}'.format(i, supp_sent_array[i]))


def answer_type_analysis():
    # data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_dev_distractor_wiki_tokenized.json')
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path, json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    yes_num = 0
    no_num = 0
    no_ans_num = 0
    span_num = 0
    for idx, row in data_frame.iterrows():
        answer = row['norm_answer']
        if answer.strip() == 'yes':
            yes_num = yes_num + 1
        elif answer.strip() == 'no':
            no_num = no_num + 1
        elif answer.strip() == 'noanswer':
            no_ans_num = no_ans_num + 1
        else:
            span_num = span_num + 1
    print('yes\t{}\nno\t{}\n no_ans\t{}\n span_num\t{}'.format(yes_num, no_num, no_ans_num, span_num))

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def data_consistent_checker():
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)
    def supp_fact_check(row):
        support_facts, filtered_support_facts = row['supporting_facts'], row['supp_facts_filtered']
        for x in support_facts:
            print('supp {}'.format(x))

        for x in filtered_support_facts:
            print('filtered supp {}'.format(x))

    def answer_check(row):
        answer_encode_id = row['answer_encode']
        answer_norm = row['norm_answer']
        orig_answer = row['answer']
        print('Decode = {}\nnorm = {}\norig = {}'.format(tokenizer.decode(answer_encode_id, skip_special_tokens=True), answer_norm, orig_answer))

    def support_sentence_checker(row):
        filtered_support_facts = row['supp_facts_filtered']
        for x in filtered_support_facts:
            print(x)
        print('=' * 100)
        p_ctx = row['p_ctx']
        for idx, context in enumerate(p_ctx):
            print(context[0])
            print('context={}\nnum sents={}'.format(context[1], len(context[1])))
            print('*'*75)
        print('+'*100)
        p_ctx_encode = row['p_ctx_encode']
        # print(len(p_ctx_encode), len(p_ctx))
        for idx, context in enumerate(p_ctx_encode):
            p_doc_encode_ids, p_doc_weight, p_doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len = context
            print('encode {}\nwith len {}\nstore len {}'.format(p_doc_encode_ids, len(p_doc_encode_ids), p_doc_len_i))
            print('sent pair = {}\nnum sents ={}'.format(sent_start_end_pair, len(sent_start_end_pair)))
            print('sent labels = {}'.format(supp_sent_labels))
            print('context len = {}'.format(len(context)))
            print('context with answer = {}'.format(ctx_with_answer))
            print('title = {}'.format(tokenizer.decode(p_doc_encode_ids[:p_title_len], skip_special_tokens=True)))
            print('answer position = {}'.format(answer_positions))
            if len(answer_positions) > 0:
                sent_start, sent_end = sent_start_end_pair[answer_positions[0][0]]
                support_sentence = tokenizer.decode(p_doc_encode_ids[sent_start:(sent_end + 1)], skip_special_tokens=True)
                print('sentence idx={}, Decode sentence = {}'.format(answer_positions[0][0], support_sentence))
                sentence_ids = p_doc_encode_ids[sent_start:(sent_end + 1)]
                decode_answer = tokenizer.decode(sentence_ids[answer_positions[0][1]:(answer_positions[0][2]+1)], skip_special_tokens=True)
                print('decode answer = {}, orig answer = {}'.format(decode_answer, row['norm_answer']))
            print(context[1])
            print('*'*75)
        print('+' * 100)

        print('p_ctx_lens', row['p_ctx_lens'])

    def doc_order_checker(row):
        pos_docs = row['p']

    '''
    _id, answer, question, supporting_facts, context, type, level, norm_query, norm_answer, p_ctx, n_ctx, supp_facts_filtered,
    answer_type, p_doc_num, n_doc_num, yes_no, no_found, ques_encode, ques_len, answer_encode, answer_len, p_ctx_encode,
    p_ctx_lens, pc_max_len, n_ctx_encode, n_ctx_lens, nc_max_len
    :return:
    '''
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path,
                              json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    print('Data frame size = {}'.format(data_frame.shape))
    record_num = data_frame.shape[0]
    row_num = 2
    random_idx = np.random.choice(record_num, row_num, replace=False)
    for idx in range(row_num):
        row_i = data_frame.loc[random_idx[idx], :]
        # supp_fact_check(row=row_i)
        # answer_check(row=row_i)
        support_sentence_checker(row=row_i)
        print('$' * 90)

def data_statistic():
    data_frame = loadWikiData(PATH=abs_distractor_wiki_path,
                              json_fileName='hotpot_train_distractor_wiki_tokenized.json')
    supp_sent_num = 0
    cand_2_sent_num = 0
    max_2_sent_num = 0
    cand_10_sent_num = 0
    max_10_sent_num = 0
    count = 0
    for idx, row in data_frame.iterrows():
        support_sents = row['supp_facts_filtered']
        sent_num_i = len(support_sents)
        p_ctxs, n_ctxs = row['p_ctx'], row['n_ctx']
        p_sent_num_i = sum([len(_[1]) for _ in p_ctxs])
        n_sent_num_i = sum([len(_[1]) for _ in n_ctxs])
        supp_sent_num = supp_sent_num + sent_num_i
        cand_2_sent_num = cand_2_sent_num + p_sent_num_i
        if max_2_sent_num < p_sent_num_i:
            max_2_sent_num = p_sent_num_i
        cand_10_sent_num = cand_10_sent_num + p_sent_num_i + n_sent_num_i
        if max_10_sent_num < (p_sent_num_i + n_sent_num_i):
            max_10_sent_num = p_sent_num_i + n_sent_num_i
        count = count + 1

    print(supp_sent_num/count, cand_2_sent_num/count, cand_10_sent_num/count)
    print(max_2_sent_num)
    print(max_10_sent_num)