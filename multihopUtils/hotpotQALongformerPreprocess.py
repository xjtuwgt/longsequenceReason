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
import string
from multihopUtils.longformerQAUtils import LongformerQATensorizer
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
from multihopUtils.longformerQAUtils import get_hotpotqa_longformer_tokenizer
import itertools
import operator
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4)
import swifter
SPECIAL_QUERY_START = '<q>' ### for query marker
SPECIAL_QUERY_END = '</q>' ### for query marker
SPECIAL_DOCUMENT_TOKEN = '<d>' ### for document maker
SPECIAL_SENTENCE_TOKEN = '<p>' ## for setence marker
CLS_TOKEN = '<s>'

def Hotpot_Train_Data_Preprocess(data: DataFrame, tokenizer: LongformerQATensorizer):
    # ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def pos_neg_context_split(row):
        question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
        doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
        supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                     if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
        positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
        norm_answer = normalize_answer(s=answer)
        if len(norm_answer) == 0:
            norm_answer = normalize_text(text=answer)
        else:
            norm_answer = normalize_text(text=norm_answer) ## normalize the answer (add a space between the answer)
        norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
        not_found_flag = False ## some answer might be not founded in supporting sentence
        ################################################################################################################
        pos_doc_num = len(positive_titles) ## number of positive documents
        pos_ctxs, neg_ctxs = [], []
        for ctx_idx, ctx in enumerate(contexts): ## Original ctx index, record the original index order
            title, text = ctx[0], ctx[1]
            text_lower = [normalize_text(text=sent) for sent in text]
            if title in positive_titles:
                count = 1
                supp_sent_flags = []
                for supp_title, supp_sent_idx in supporting_facts_filtered:
                    if title == supp_title:
                        supp_sent = text_lower[supp_sent_idx]
                        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
                            has_answer = answer_span_checker(norm_answer.strip(), supp_sent)
                            if has_answer:
                                encode_has_answer, _, _ = find_answer_span(norm_answer.strip(), supp_sent, tokenizer)
                                if not encode_has_answer:
                                    encode_has_answer,  X, Y = find_answer_span(norm_answer, supp_sent, tokenizer)
                                    if not encode_has_answer:
                                        supp_sent_flags.append((supp_sent_idx, False))
                                        # print('answer {} || sent {}'.format(norm_answer, supp_sent))
                                    else:
                                        supp_sent_flags.append((supp_sent_idx, True))
                                        count = count + 1
                                else:
                                    supp_sent_flags.append((supp_sent_idx, True))
                                    count = count + 1
                            else:
                                supp_sent_flags.append((supp_sent_idx, False))
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                pos_ctxs.append([title.lower(), text_lower, count, supp_sent_flags, ctx_idx])  ## Identify the support document with answer
            else:
                neg_ctxs.append([title.lower(), text_lower, 0, [], ctx_idx])
        neg_doc_num = len(neg_ctxs)
        pos_counts = [x[2] for x in pos_ctxs]
        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
            if sum(pos_counts) == 2:
                not_found_flag = True
        assert len(pos_counts) == 2
        if not_found_flag:
            norm_answer = 'noanswer'
        if (pos_counts[0] > 1 and pos_counts[1] > 1) or (pos_counts[0] <= 1 and pos_counts[1] <= 1):
            supp_doc_type = False
        else:
            supp_doc_type = True
        yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
        return norm_question, norm_answer, pos_ctxs, neg_ctxs, supporting_facts_filtered, supp_doc_type, pos_doc_num, neg_doc_num, yes_no_flag, not_found_flag
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_answer', 'p_ctx', 'n_ctx', 'supp_facts_filtered', 'p_doc_type', 'p_doc_num',
          'n_doc_num', 'yes_no', 'no_found']] = data.swifter.apply(lambda row: pd.Series(pos_neg_context_split(row)), axis=1)
    not_found_num = data[data['no_found']].shape[0]
    print('Splitting positive samples from negative samples takes {:.4f} seconds, answer not found = {}'.format(time() - start_time, not_found_num))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, pos_ctxs, neg_ctxs, norm_answer = row['norm_query'], row['p_ctx'], row['n_ctx'], row['norm_answer']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_encode_ids = query_encoder(query=norm_question, tokenizer=tokenizer)
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:
            answer_encode_ids = [-1]
        else:
            answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
        answer_len = len(answer_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        positive_ctx_encode_ids = []
        max_positive_doc_len = 0 ##max ctx len
        positive_ctx_lens = [] ## positive ctx lens
        for p_idx, content in enumerate(pos_ctxs):
            p_title, p_doc, p_doc_weight, supp_sent_flags, _ = content
            p_doc_encode_ids, sent_start_end_pair, p_doc_len_i, p_title_len = document_encoder(title=p_title, doc_sents=p_doc, tokenizer=tokenizer)
            # #########################
            if max_positive_doc_len < p_doc_len_i:
                max_positive_doc_len = p_doc_len_i
            #########################
            assert len(p_doc) == len(sent_start_end_pair)
            positive_ctx_lens.append(p_doc_len_i)
            supp_sent_labels = [0] * len(p_doc) ## number of sentences in postive document
            ctx_with_answer = False
            answer_positions = [] ## answer position
            for sup_sent_idx, supp_sent_flag in supp_sent_flags:
                supp_sent_labels[sup_sent_idx] = 1 ## support sentence without answer
                if supp_sent_flag:
                    start_id, end_id = sent_start_end_pair[sup_sent_idx]
                    supp_sent_labels[sup_sent_idx] = 2 ## support sentence with answer
                    supp_sent_encode_ids = p_doc_encode_ids[start_id:(end_id+1)]
                    #answer_start_idx = find_sub_list(target=answer_encode_ids, source=supp_sent_encode_ids)
                    ####################################################################################################
                    answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
                    answer_start_idx = sub_list_extact_match_idx(target=answer_encode_ids, source=supp_sent_encode_ids)
                    if answer_start_idx < 0:
                        answer_encode_ids = tokenizer.text_encode(text=norm_answer.strip(), add_special_tokens=False)
                        answer_start_idx = sub_list_extact_match_idx(target=answer_encode_ids, source=supp_sent_encode_ids)
                    answer_len = len(answer_encode_ids)
                    ####################################################################################################
                    assert answer_start_idx > 0, "supp sent {} \n answer={} \n p_doc = {} \n answer={} \n {} \n {}".format(tokenizer.tokenizer.decode(supp_sent_encode_ids),
                                                                                     tokenizer.tokenizer.decode(answer_encode_ids), p_doc[sup_sent_idx], norm_answer, supp_sent_encode_ids, answer_encode_ids)
                    ctx_with_answer = True ## support sentence with answer
                    answer_positions.append((sup_sent_idx, answer_start_idx, answer_start_idx + answer_len - 1)) # supp sent idx, relative start, relative end
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            p_tuple = (p_doc_encode_ids, p_doc_weight, p_doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len)
            positive_ctx_encode_ids.append(p_tuple)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        negative_ctx_encode_ids = []
        negative_ctx_lens = []
        max_negative_doc_len = 0
        for n_idx, content in enumerate(neg_ctxs):
            n_title, n_doc, n_doc_weight, _, _ = content
            n_doc_encode_ids, sent_start_end_pair, n_doc_len_i, n_title_len = document_encoder(title=n_title, doc_sents=n_doc, tokenizer=tokenizer)
            negative_ctx_lens.append(n_doc_len_i)
            if max_negative_doc_len < n_doc_len_i:
                max_negative_doc_len = n_doc_len_i
            n_tuple = (n_doc_encode_ids, n_doc_weight, n_doc_len_i, sent_start_end_pair, [0] * len(n_doc), False, [], n_title_len)
            negative_ctx_encode_ids.append(n_tuple)
        return query_encode_ids, query_len, answer_encode_ids, answer_len, positive_ctx_encode_ids, positive_ctx_lens, max_positive_doc_len, \
               negative_ctx_encode_ids, negative_ctx_lens, max_negative_doc_len
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['ques_encode', 'ques_len', 'answer_encode', 'answer_len', 'p_ctx_encode', 'p_ctx_lens', 'pc_max_len',
          'n_ctx_encode', 'n_ctx_lens', 'nc_max_len']] = \
        data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    return data

####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Hotpot_Dev_Data_Preprocess(data: DataFrame, tokenizer: LongformerQATensorizer):
    # ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def pos_neg_context_split(row):
        question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
        doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
        supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                     if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
        positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
        norm_answer = normalize_answer(s=answer)
        if len(norm_answer) == 0:
            norm_answer = normalize_text(text=answer)
        else:
            norm_answer = normalize_text(text=norm_answer) ## normalize the answer (add a space between the answer)
        norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
        not_found_flag = False ## some answer might be not founded in supporting sentence
        ################################################################################################################
        pos_doc_num = len(positive_titles) ## number of positive documents
        pos_ctxs, neg_ctxs = [], []
        for ctx_idx, ctx in enumerate(contexts): ## Original ctx index, record the original index order
            title, text = ctx[0], ctx[1]
            text_lower = [normalize_text(text=sent) for sent in text]
            if title in positive_titles:
                count = 1
                supp_sent_flags = []
                for supp_title, supp_sent_idx in supporting_facts_filtered:
                    if title == supp_title:
                        supp_sent = text_lower[supp_sent_idx]
                        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
                            has_answer = answer_span_checker(norm_answer.strip(), supp_sent)
                            if has_answer:
                                encode_has_answer, _, _ = find_answer_span(norm_answer.strip(), supp_sent, tokenizer)
                                if not encode_has_answer:
                                    encode_has_answer, X, Y = find_answer_span(norm_answer, supp_sent, tokenizer)
                                    if not encode_has_answer:
                                        supp_sent_flags.append((supp_sent_idx, False))
                                        # print('answer {} || sent {}'.format(norm_answer, supp_sent))
                                    else:
                                        supp_sent_flags.append((supp_sent_idx, True))
                                        count = count + 1
                                else:
                                    supp_sent_flags.append((supp_sent_idx, True))
                                    count = count + 1
                            else:
                                supp_sent_flags.append((supp_sent_idx, False))
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                pos_ctxs.append([title.lower(), text_lower, count, supp_sent_flags, ctx_idx])  ## Identify the support document with answer
            else:
                neg_ctxs.append([title.lower(), text_lower, 0, [], ctx_idx])
        neg_doc_num = len(neg_ctxs)
        pos_counts = [x[2] for x in pos_ctxs]
        if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
            if sum(pos_counts) == 2:
                not_found_flag = True
        assert len(pos_counts) == 2
        if not_found_flag:
            norm_answer = 'noanswer'
        if (pos_counts[0] > 1 and pos_counts[1] > 1) or (pos_counts[0] <= 1 and pos_counts[1] <= 1):
            supp_doc_type = False
        else:
            supp_doc_type = True
        yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
        return norm_question, norm_answer, pos_ctxs, neg_ctxs, supporting_facts_filtered, supp_doc_type, pos_doc_num, neg_doc_num, yes_no_flag, not_found_flag
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_answer', 'p_ctx', 'n_ctx', 'supp_facts_filtered', 'p_doc_type', 'p_doc_num',
          'n_doc_num', 'yes_no', 'no_found']] = data.swifter.apply(lambda row: pd.Series(pos_neg_context_split(row)), axis=1)
    not_found_num = data[data['no_found']].shape[0]
    print('Splitting positive samples from negative samples takes {:.4f} seconds, answer not found = {}'.format(time() - start_time, not_found_num))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, pos_ctxs, neg_ctxs, norm_answer = row['norm_query'], row['p_ctx'], row['n_ctx'], row['norm_answer']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_encode_ids = query_encoder(query=norm_question, tokenizer=tokenizer)
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if norm_answer.strip() in ['yes', 'no', 'noanswer']:
            answer_encode_ids = [-1]
        else:
            answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
        answer_len = len(answer_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        positive_ctx_encode_ids = []
        max_positive_doc_len = 0 ##max ctx len
        positive_ctx_lens = [] ## positive ctx lens
        for p_idx, content in enumerate(pos_ctxs):
            p_title, p_doc, p_doc_weight, supp_sent_flags, _ = content
            p_doc_encode_ids, sent_start_end_pair, p_doc_len_i, p_title_len = document_encoder(title=p_title, doc_sents=p_doc, tokenizer=tokenizer)
            # #########################
            if max_positive_doc_len < p_doc_len_i:
                max_positive_doc_len = p_doc_len_i
            #########################
            assert len(p_doc) == len(sent_start_end_pair)
            positive_ctx_lens.append(p_doc_len_i)
            supp_sent_labels = [0] * len(p_doc) ## number of sentences in postive document
            ctx_with_answer = False
            answer_positions = [] ## answer position
            for sup_sent_idx, supp_sent_flag in supp_sent_flags:
                supp_sent_labels[sup_sent_idx] = 1 ## support sentence without answer
                if supp_sent_flag:
                    start_id, end_id = sent_start_end_pair[sup_sent_idx]
                    supp_sent_labels[sup_sent_idx] = 2 ## support sentence with answer
                    supp_sent_encode_ids = p_doc_encode_ids[start_id:(end_id+1)]
                    #answer_start_idx = find_sub_list(target=answer_encode_ids, source=supp_sent_encode_ids)
                    ####################################################################################################
                    answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
                    answer_start_idx = sub_list_extact_match_idx(target=answer_encode_ids, source=supp_sent_encode_ids)
                    if answer_start_idx < 0:
                        answer_encode_ids = tokenizer.text_encode(text=norm_answer.strip(), add_special_tokens=False)
                        answer_start_idx = sub_list_extact_match_idx(target=answer_encode_ids, source=supp_sent_encode_ids)
                    answer_len = len(answer_encode_ids)
                    ####################################################################################################
                    assert answer_start_idx > 0, "supp sent {} \n answer={} \n p_doc = {} \n answer={} \n {} \n {}".format(tokenizer.tokenizer.decode(supp_sent_encode_ids),
                                                                                     tokenizer.tokenizer.decode(answer_encode_ids), p_doc[sup_sent_idx], norm_answer, supp_sent_encode_ids, answer_encode_ids)
                    ctx_with_answer = True ## support sentence with answer
                    answer_positions.append((sup_sent_idx, answer_start_idx, answer_start_idx + answer_len - 1)) # supp sent idx, relative start, relative end
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            p_tuple = (p_doc_encode_ids, p_doc_weight, p_doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, p_title_len)
            positive_ctx_encode_ids.append(p_tuple)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        negative_ctx_encode_ids = []
        negative_ctx_lens = []
        max_negative_doc_len = 0
        for n_idx, content in enumerate(neg_ctxs):
            n_title, n_doc, n_doc_weight, _, _ = content
            n_doc_encode_ids, sent_start_end_pair, n_doc_len_i, n_title_len = document_encoder(title=n_title, doc_sents=n_doc, tokenizer=tokenizer)
            negative_ctx_lens.append(n_doc_len_i)
            if max_negative_doc_len < n_doc_len_i:
                max_negative_doc_len = n_doc_len_i
            n_tuple = (n_doc_encode_ids, n_doc_weight, n_doc_len_i, sent_start_end_pair, [0] * len(n_doc), False, [], n_title_len)
            negative_ctx_encode_ids.append(n_tuple)
        return query_encode_ids, query_len, answer_encode_ids, answer_len, positive_ctx_encode_ids, positive_ctx_lens, max_positive_doc_len, \
               negative_ctx_encode_ids, negative_ctx_lens, max_negative_doc_len
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['ques_encode', 'ques_len', 'answer_encode', 'answer_len', 'p_ctx_encode', 'p_ctx_lens', 'pc_max_len',
          'n_ctx_encode', 'n_ctx_lens', 'nc_max_len']] = \
        data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    return data

#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Hotpot_Test_Data_PreProcess(data: DataFrame, tokenizer: LongformerQATensorizer):
    # focus on ['question', 'context', '_id']
    """
    Supporting_facts: pair of (title, sentence index) --> (str, int)
    Level: {easy, medium, hard}
    Question: query --> str
    Context: list of pair (title, text) --> (str, list(str))
    Answer: str
    Type: {comparison, bridge}
    """
    def norm_context(row):
        question, contexts = row['question'], row['context']
        norm_question = normalize_question(question.lower())
        hotpot_ctxs = []
        ################################################################################################################
        for ctx_idx, ctx in enumerate(contexts):  ## Original ctx index
            title, text = ctx[0], ctx[1]
            text_lower = [normalize_text(text=sent) for sent in text]
            hotpot_ctxs.append([title.lower(), text_lower, ctx_idx])
        return norm_question, hotpot_ctxs
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['norm_query', 'norm_ctx']] = data.swifter.apply(lambda row: pd.Series(norm_context(row)), axis=1)
    print('Normalizing samples takes {:.4f} seconds'.format(time() - start_time))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, norm_ctxs = row['norm_query'], row['norm_ctx']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_encode_ids = query_encoder(query=norm_question, tokenizer=tokenizer)
        query_len = len(query_encode_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_encode_ids = []
        max_doc_len = 0
        ctx_lens = []
        for ctx_idx, content in enumerate(norm_ctxs):
            title, doc_sents, _ = content
            doc_encode_ids, sent_start_end_pair, doc_len_i, title_len = document_encoder(title=title, doc_sents=doc_sents, tokenizer=tokenizer)
            # #########################
            if max_doc_len < doc_len_i:
                max_doc_len = doc_len_i
            #########################
            assert len(doc_sents) == len(sent_start_end_pair)
            ctx_lens.append(doc_len_i)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_tuple = (doc_encode_ids, doc_len_i, sent_start_end_pair, title_len, ctx_idx)
            ctx_encode_ids.append(ctx_tuple)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return query_encode_ids, query_len, ctx_encode_ids, ctx_lens, max_doc_len
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    data[['ques_encode', 'ques_len', 'ctx_encode', 'ctx_lens', 'ctx_max_len']] = \
        data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    return data
#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def query_encoder(query: str, tokenizer: LongformerQATensorizer):
    query_res = CLS_TOKEN + SPECIAL_QUERY_START + query + SPECIAL_QUERY_END
    query_encode_ids = tokenizer.text_encode(text=query_res, add_special_tokens=False)
    return query_encode_ids

def document_encoder(title: str, doc_sents: list, tokenizer: LongformerQATensorizer):
    title_res = SPECIAL_DOCUMENT_TOKEN + title ##
    title_encode_ids = tokenizer.text_encode(text=title_res, add_special_tokens=False)
    title_len = len(title_encode_ids)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encode_id_lens = []
    encode_id_lens.append(title_len)
    doc_encode_id_list = []
    doc_encode_id_list.append(title_encode_ids)
    for sent_idx, sent_text in enumerate(doc_sents):
        sent_text_res = SPECIAL_SENTENCE_TOKEN + sent_text
        sent_encode_ids = tokenizer.text_encode(text=sent_text_res, add_special_tokens=False)
        doc_encode_id_list.append(sent_encode_ids)
        sent_len = len(sent_encode_ids)
        encode_id_lens.append(sent_len)
    doc_sent_len_cum_list = list(itertools.accumulate(encode_id_lens, operator.add))
    sent_start_end_pair = [(doc_sent_len_cum_list[i], doc_sent_len_cum_list[i + 1] - 1) for i in range(len(encode_id_lens) - 1)]
    doc_encode_ids = list(itertools.chain.from_iterable(doc_encode_id_list))
    assert len(doc_encode_ids) == doc_sent_len_cum_list[-1]
    return doc_encode_ids, sent_start_end_pair, len(doc_encode_ids), title_len

def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question

def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip() ###adding the ' ' is important to make the consist encoder, for roberta tokenizer
    return text

def normalize_answer(s):
    def remove_punc(text):
        return text.strip(string.punctuation)
    def lower(text):
        return text.lower()
    return remove_punc(lower(s))

def answer_span_checker(answer, sentence):
    find_idx = sentence.find(answer)
    return find_idx >=0

def find_answer_span(norm_answer, sentence, tokenizer):
    answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
    sentence_encode_ids = tokenizer.text_encode(text=sentence, add_special_tokens=False)
    idx = sub_list_extact_match_idx(target=answer_encode_ids, source=sentence_encode_ids)
    flag = idx >= 0
    return flag, answer_encode_ids, sentence_encode_ids

def sub_list_extact_match_idx(target: list, source: list) -> int:
    idx = find_sub_list(target, source)
    return idx

def find_sub_list(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    def equal_list(a_list, b_list):
        for j in range(len(a_list)):
            if a_list[j] != b_list[j]:
                return False
        return True
    for i in range(len(source) - len(target) + 1):
        temp = source[i:(i+t_len)]
        is_equal = equal_list(target, temp)
        if is_equal:
            return i
    return -1
########################################################################################################################
def hotpotqa_preprocess_example():
    start_time = time()
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE)
    longformer_tokenizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=-1)
    dev_data, _ = HOTPOT_DevData_Distractor()
    print('*' * 75)
    dev_test_data = Hotpot_Test_Data_PreProcess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev-test records'.format(dev_test_data.shape[0]))
    dev_test_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_test_distractor_wiki_tokenized.json'))
    print('*' * 75)
    dev_data, _ = HOTPOT_DevData_Distractor()
    dev_data = Hotpot_Dev_Data_Preprocess(data=dev_data, tokenizer=longformer_tokenizer)
    print('Get {} dev records'.format(dev_data.shape[0]))
    dev_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_dev_distractor_wiki_tokenized.json'))
    print('*' * 75)
    train_data, _ = HOTPOT_TrainData()
    train_data = Hotpot_Train_Data_Preprocess(data=train_data, tokenizer=longformer_tokenizer)
    print('Get {} training records'.format(train_data.shape[0]))
    train_data.to_json(os.path.join(abs_distractor_wiki_path, 'hotpot_train_distractor_wiki_tokenized.json'))
    print('Runtime = {:.4f} seconds'.format(time() - start_time))
    print('*' * 75)

if __name__ == '__main__':
    hotpotqa_preprocess_example()
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