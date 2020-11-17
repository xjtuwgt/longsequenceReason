from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.linalg import block_diag
from multihopUtils.longformerQAUtils import LongformerQATensorizer

def position_filter(start_end_pair_list: list, max_len: int):
    filtered_positions = []
    filtered_lens = []
    for pos_pair in start_end_pair_list:
        p_st, p_en = pos_pair
        if p_st >= max_len:
            p_st, p_en = 0, 0
        else:
            p_st = p_st
            p_en = max_len - 1 if p_en >= max_len else p_en
        if p_en == 0:
            filtered_lens.append(0)
        else:
            filtered_lens.append(p_en - p_st + 1)
        filtered_positions.append((p_st, p_en))
    return filtered_positions, filtered_lens

def mask_generation(sent_num_docs: list, max_sent_num: int):
    assert len(sent_num_docs) > 0 and sent_num_docs[0] > 0
    ss_attn_mask = np.ones((sent_num_docs[0], sent_num_docs[0]))
    sd_attn_mask = np.ones((1, sent_num_docs[0]))
    doc_pad_num = 0
    for idx in range(1, len(sent_num_docs)):
        sent_num_i = sent_num_docs[idx]
        if sent_num_i > 0:
            ss_mask_i = np.ones((sent_num_i, sent_num_i))
            ss_attn_mask = block_diag(ss_attn_mask, ss_mask_i)
            sd_mask_i = np.ones((1, sent_num_i))
            sd_attn_mask = block_diag(sd_attn_mask, sd_mask_i)
        else:
            doc_pad_num = doc_pad_num + 1

    sent_num_sum = sum(sent_num_docs)
    assert sent_num_sum <= max_sent_num, '{}, max {}'.format(sent_num_sum, max_sent_num)
    ss_attn_mask = torch.from_numpy(ss_attn_mask).type(torch.bool)
    sd_attn_mask = torch.from_numpy(sd_attn_mask).type(torch.bool)
    sent_pad_num = max_sent_num - sent_num_sum
    if sent_pad_num > 0:
        ss_attn_mask = F.pad(ss_attn_mask, [0, sent_pad_num, 0, sent_pad_num], 'constant', False)
        sd_attn_mask = F.pad(sd_attn_mask, [0, sent_pad_num, 0, 0], 'constant', False)
    if doc_pad_num > 0:
        sd_attn_mask = F.pad(sd_attn_mask, [0, 0, 0, doc_pad_num], 'constant', False)
    return ss_attn_mask, sd_attn_mask

####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class HotpotTrainDataset(Dataset): ##for training data loader
    def __init__(self, data_frame: DataFrame, hotpot_tensorizer: LongformerQATensorizer,
                 max_doc_num=2, max_sent_num=150, training_shuffle=False,
                 global_mask_type: str = 'query_doc_sent'):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = hotpot_tensorizer.max_length
        self.hotpot_tensorizer = hotpot_tensorizer
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.global_mask_type = global_mask_type # query, query_doc, query_doc_sent
        self.training_shuffle = training_shuffle

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens = example['p_ctx_encode'], example['p_ctx_lens']
        norm_answer = example['norm_answer']
        yes_no_question = False
        if norm_answer.strip() in ['yes', 'no', 'noanswer']: ## yes: 1, no/noanswer: 2, span = 0
            yes_no_label = torch.LongTensor([1]) if norm_answer.strip() == 'yes' else torch.LongTensor([2])
            yes_no_question = True
        else:
            yes_no_label = torch.LongTensor([0]) ## span answer = 0
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        not_found_answer = example['no_found']
        if not_found_answer:
            yes_no_question = True
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pos_position = [_[4] for _ in example['p_ctx']]
        pos_weights = [_[2] for _ in example['p_ctx']]
        ## ++++++++++++++++++++++++++++++++++++++++
        ctx_enocde = pos_ctx_encode
        ctx_lens = pos_ctx_lens
        doc_labels = [1] * len(pos_ctx_encode)
        ctx_weights = pos_weights
        ctx_orig_position = pos_position
        doc_num = len(doc_labels)
        assert len(ctx_enocde) == len(ctx_lens) and len(ctx_enocde) == len(doc_labels) and doc_num >=2
        ##++++++++++++++++++++++++++++++++
        if not self.training_shuffle:
            orig_orders = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(ctx_orig_position))]
            ctx_enocde = [ctx_enocde[orig_orders[i]] for i in range(doc_num)]
            ctx_lens = [ctx_lens[orig_orders[i]] for i in range(doc_num)]
            doc_labels = [doc_labels[orig_orders[i]] for i in range(doc_num)]
            ctx_weights = [ctx_weights[orig_orders[i]] for i in range(doc_num)]
        ##++++++++++++++++++++++++++++++++
        else:
            shuffle_ctx_index = np.random.choice(doc_num, doc_num, replace=False)
            ctx_enocde = [ctx_enocde[shuffle_ctx_index[i]] for i in range(doc_num)]
            ctx_lens = [ctx_lens[shuffle_ctx_index[i]] for i in range(doc_num)]
            doc_labels = [doc_labels[shuffle_ctx_index[i]] for i in range(doc_num)]
            ctx_weights = [ctx_weights[shuffle_ctx_index[i]] for i in range(doc_num)]
        ##++++++++++++++++++++++++++++++++
        ################################################################################################################
        ctx_label_weight = [(i, doc_labels[i], ctx_weights[i]) for i in range(len(doc_labels)) if doc_labels[i] == 1]
        assert len(ctx_label_weight) == 2
        if ctx_label_weight[0][2] < ctx_label_weight[1][2]:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[0][0]]), torch.LongTensor([ctx_label_weight[1][0]])
        else:
            head_doc_idx, tail_doc_idx = torch.LongTensor([ctx_label_weight[1][0]]), torch.LongTensor([ctx_label_weight[0][0]])
        ################################################################################################################
        ################################################################################################################
        concat_encode = query_encode ## concat encode ids (query + 10 documents)
        concat_len = query_len ## concat encode ids length (query + 10 documents)
        concat_sent_num = 0  # compute the number of sentences
        doc_start_end_pair_list = []
        sent_start_end_pair_list = []
        ctx_sent_lens = [] ## number of sentences, the token of each sentence
        ###############################
        doc_sent_nums = [] # the number of sentences in each document = number of document
        ctx_sent2doc_map_list = [] ## length is equal to sent numbers
        ctx_sentIndoc_idx_list = [] ## length is equal to sent numbers
        ###############################
        supp_sent_labels_list = [] # equal to number of sentences
        answer_position_list = []
        previous_len = query_len
        ###############################
        ctx_title_lens = []
        ###############################
        for doc_idx, doc_tup in enumerate(ctx_enocde):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, title_len = doc_tup
            ##############################################
            concat_sent_num = concat_sent_num + len(sent_start_end_pair)
            doc_sent_nums = doc_sent_nums + [len(sent_start_end_pair)] ## number of sentences
            # =======================================
            ctx_sent2doc_map_list = ctx_sent2doc_map_list + [doc_idx] * len(sent_start_end_pair) ## sentence to doc idx
            ctx_sentIndoc_idx_list = ctx_sentIndoc_idx_list + [x for x in range(len(sent_start_end_pair))] ## sentence to original sent index
            # =======================================
            ctx_sent_lens = ctx_sent_lens + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
            # =======================================
            # +++++++++++++++++++++++++++++++++++++++
            ctx_title_lens.append(title_len)
            # +++++++++++++++++++++++++++++++++++++++
            assert len(doc_encode_ids) == ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i \
                   and doc_len_i == ctx_lens[doc_idx] and len(sent_start_end_pair) == len(supp_sent_labels)
            concat_encode = concat_encode + doc_encode_ids
            concat_len = concat_len + doc_len_i
            # =======================================
            doc_start_end_pair_list.append((previous_len, previous_len + doc_len_i - 1))
            sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
            sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
            # =======================================
            supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
            if len(answer_positions) > 0:
                for a_idx, answer_pos in enumerate(answer_positions):
                    sent_a_idx, a_start, a_end = answer_pos
                    sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                    temp_position = (sent_off_set + a_start, sent_off_set + a_end)
                    answer_position_list.append(temp_position)
            previous_len = previous_len + doc_len_i
        # ==============================================================================
        ctx_token2sent_map_list = [0] * concat_len
        for sent_idx, sent_position in enumerate(sent_start_end_pair_list):
            ctx_token2sent_map_list[sent_position[0]:(sent_position[1] + 1)] = [sent_idx] * (ctx_sent_lens[sent_idx])
        # ==============================================================================
        assert doc_start_end_pair_list[-1][1] + 1 == concat_len
        assert len(ctx_token2sent_map_list) == concat_len, '{}, {}'.format(len(ctx_token2sent_map_list), concat_len)
        assert len(doc_labels) == len(doc_start_end_pair_list)
        assert previous_len == len(concat_encode) and previous_len == concat_len
        assert len(ctx_sent2doc_map_list) == len(supp_sent_labels_list)
        assert len(ctx_sentIndoc_idx_list) == len(supp_sent_labels_list)
        assert len(supp_sent_labels_list) == len(sent_start_end_pair_list) and concat_sent_num == len(sent_start_end_pair_list)
        doc_start_end_pair_list, ctx_lens = position_filter(doc_start_end_pair_list, max_len=self.max_token_num)
        sent_start_end_pair_list, ctx_sent_lens = position_filter(sent_start_end_pair_list, max_len=self.max_token_num)
        # print('ctx after', ctx_lens)
        if doc_num < self.max_doc_num:
            pad_doc_num = self.max_doc_num - doc_num
            doc_start_end_pair_list = doc_start_end_pair_list + [(0, 0)] * pad_doc_num
            ctx_lens = ctx_lens + [0] * pad_doc_num
            doc_labels = doc_labels + [0] * pad_doc_num
            doc_sent_nums = doc_sent_nums + [0] * pad_doc_num
            # ================================================
            ctx_title_lens = ctx_title_lens + [0] * pad_doc_num
            # ================================================
        ###############################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ###############################################################################################################
        if concat_sent_num < self.max_sent_num:
            pad_sent_num = self.max_sent_num - concat_sent_num
            sent_start_end_pair_list = sent_start_end_pair_list + [(0, 0)] * pad_sent_num
            ctx_sent_lens = ctx_sent_lens + [0] * pad_sent_num
            supp_sent_labels_list = supp_sent_labels_list + [0] * pad_sent_num
            #########+++++++++++++++++++++++++++++++++++
            ctx_sent2doc_map_list = ctx_sent2doc_map_list + [0] * pad_sent_num
            ctx_sentIndoc_idx_list = ctx_sentIndoc_idx_list + [0] * pad_sent_num
            #########+++++++++++++++++++++++++++++++++++
        if not yes_no_question:
            answer_position = answer_position_list[0]
        else:
            answer_position = (0, 0)

        if len(ctx_token2sent_map_list) < self.max_token_num:
            pad_token_num = self.max_token_num - len(ctx_token2sent_map_list)
            ctx_token2sent_map_list = ctx_token2sent_map_list + [0] * pad_token_num

        cat_doc_encodes = self.hotpot_tensorizer.token_ids_to_tensor(token_ids=concat_encode)
        cat_doc_attention_mask = self.hotpot_tensorizer.get_attn_mask(token_ids_tensor=cat_doc_encodes)
        if self.global_mask_type == 'query':
            query_mask_idxes = [x for x in range(query_len)]
        elif self.global_mask_type == 'query_doc':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list]
        elif self.global_mask_type == 'query_doc_sent':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[0] for x in sent_start_end_pair_list]
        else:
            query_mask_idxes = [x for x in range(query_len)]
        cat_doc_global_attn_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes, gobal_mask_idxs=query_mask_idxes)
        #######
        marker_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[0] for x in sent_start_end_pair_list]
        # +++++++++++++++++++++++++++++++++++++++++++++++++
        title_mask_idxes = []
        for t_idx, ti_len in enumerate(ctx_title_lens):
            if ti_len > 0:
                title_start_idx = doc_start_end_pair_list[t_idx][0] + 1
                for j in range(ti_len):
                    title_mask_idxes.append(title_start_idx + j)
        marker_mask_idxes = marker_mask_idxes + title_mask_idxes
        # +++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_marker_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes, gobal_mask_idxs=marker_mask_idxes)
        ctx_marker_mask = ctx_marker_mask.type(torch.bool)
        #######
        doc_start_idxes = torch.LongTensor([x[0] for x in doc_start_end_pair_list])
        doc_end_idxes = torch.LongTensor([x[1] for x in doc_start_end_pair_list])
        sent_start_idxes = torch.LongTensor([x[0] for x in sent_start_end_pair_list])
        sent_end_idxes = torch.LongTensor([x[1] for x in sent_start_end_pair_list])
        answer_start_idx, answer_end_idx = torch.LongTensor([answer_position[0]]), torch.LongTensor([answer_position[1]])
        doc_lens = torch.LongTensor(ctx_lens)
        doc_labels = torch.LongTensor(doc_labels)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map_list)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx_list)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert concat_len <= self.max_token_num
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               answer_start_idx, answer_end_idx, doc_lens, doc_labels, ctx_sent_lens, supp_sent_labels, yes_no_label, head_doc_idx, \
               tail_doc_idx, ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[22] for _ in data])
        batch_max_sent_num = max([_[23] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[6] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[7] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[8] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[11] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_head_idx = torch.stack([_[12] for _ in data], dim=0)
        batch_tail_idx = torch.stack([_[13] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[14] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[15] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[16] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[17] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[18] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[19] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[20] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[21] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
                'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'head_idx': batch_head_idx,
               'tail_idx': batch_tail_idx, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker, 't2s_map': batch_token2sent}
        return res
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##**********************************************************************************************************************
##**********************************************************************************************************************
class HotpotDevDataset(Dataset): ##for dev dataloader
    def __init__(self, data_frame: DataFrame, hotpot_tensorizer: LongformerQATensorizer,
                 max_doc_num=2, max_sent_num=150, global_mask_type: str = 'query_doc_sent'):
        self.len = data_frame.shape[0]
        self.data = data_frame
        self.max_token_num = hotpot_tensorizer.max_length
        self.hotpot_tensorizer = hotpot_tensorizer
        self.max_doc_num = max_doc_num
        self.max_sent_num = max_sent_num
        self.global_mask_type = global_mask_type # query, query_doc, query_doc_sent

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        query_encode, query_len = example['ques_encode'], example['ques_len']
        pos_ctx_encode, pos_ctx_lens= example['p_ctx_encode'], example['p_ctx_lens']
        norm_answer = example['norm_answer']
        if norm_answer.strip() in ['yes', 'no', 'noanswer']: ## yes: 1, no/noanswer: 2, span = 0
            yes_no_label = torch.LongTensor([1]) if norm_answer.strip() == 'yes' else torch.LongTensor([2])
        else:
            yes_no_label = torch.LongTensor([0])
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        yes_no_question = example['yes_no']
        not_found_answer = example['no_found']
        if not_found_answer:
            yes_no_question = True
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_enocde = pos_ctx_encode
        ctx_lens = pos_ctx_lens
        doc_labels = [1] * len(pos_ctx_encode)
        doc_num = len(doc_labels)
        pos_position = [_[4] for _ in example['p_ctx']]
        ctx_orig_position = pos_position
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        orig_orders = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(ctx_orig_position))]
        ctx_enocde = [ctx_enocde[orig_orders[i]] for i in range(doc_num)]
        ctx_lens = [ctx_lens[orig_orders[i]] for i in range(doc_num)]
        doc_labels = [doc_labels[orig_orders[i]] for i in range(doc_num)]
        ####################################################
        ####################################################
        concat_encode = query_encode  ## concat encode ids (query + 10 documents)
        concat_len = query_len  ## concat encode ids length (query + 10 documents)
        concat_sent_num = 0  # compute the number of sentences
        doc_start_end_pair_list = []
        sent_start_end_pair_list = []
        ctx_sent_lens = []  ## number of sentences, the token of each sentence
        ###############################
        doc_sent_nums = []  # the number of sentences in each document = number of document
        ctx_sent2doc_map_list = []  ## length is equal to sent numbers
        ctx_sentIndoc_idx_list = []  ## length is equal to sent numbers
        ###############################
        supp_sent_labels_list = []  # equal to number of sentences
        answer_position_list = []
        previous_len = query_len
        ###############################
        ctx_title_lens = []
        ###############################
        for doc_idx, doc_tup in enumerate(ctx_enocde):
            doc_encode_ids, doc_weight, doc_len_i, sent_start_end_pair, supp_sent_labels, ctx_with_answer, answer_positions, title_len = doc_tup
            ##############################################
            concat_sent_num = concat_sent_num + len(sent_start_end_pair)
            doc_sent_nums = doc_sent_nums + [len(sent_start_end_pair)]  ## number of sentences
            # =======================================
            ctx_sent2doc_map_list = ctx_sent2doc_map_list + [doc_idx] * len(sent_start_end_pair)  ## sentence to doc idx
            ctx_sentIndoc_idx_list = ctx_sentIndoc_idx_list + [x for x in range(
                len(sent_start_end_pair))]  ## sentence to original sent index
            # =======================================
            ctx_sent_lens = ctx_sent_lens + [x[1] - x[0] + 1 for x in sent_start_end_pair if x[1] > 0]
            # =======================================
            # +++++++++++++++++++++++++++++++++++++++
            ctx_title_lens.append(title_len)
            # +++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++
            assert len(doc_encode_ids) == ctx_lens[doc_idx] and len(doc_encode_ids) == doc_len_i \
                   and doc_len_i == ctx_lens[doc_idx] and len(sent_start_end_pair) == len(supp_sent_labels)
            concat_encode = concat_encode + doc_encode_ids
            concat_len = concat_len + doc_len_i
            # =======================================
            doc_start_end_pair_list.append((previous_len, previous_len + doc_len_i - 1))
            sent_start_end_pair_i = [(x[0] + previous_len, x[1] + previous_len) for x in sent_start_end_pair]
            sent_start_end_pair_list = sent_start_end_pair_list + sent_start_end_pair_i
            # =======================================
            supp_sent_labels_list = supp_sent_labels_list + supp_sent_labels
            if len(answer_positions) > 0:
                for a_idx, answer_pos in enumerate(answer_positions):
                    sent_a_idx, a_start, a_end = answer_pos
                    sent_off_set = sent_start_end_pair_i[sent_a_idx][0]
                    temp_position = (sent_off_set + a_start, sent_off_set + a_end)
                    answer_position_list.append(temp_position)
            previous_len = previous_len + doc_len_i
        # ==============================================================================
        ctx_token2sent_map_list = [0] * concat_len
        for sent_idx, sent_position in enumerate(sent_start_end_pair_list):
            ctx_token2sent_map_list[sent_position[0]:(sent_position[1] + 1)] = [sent_idx] * (ctx_sent_lens[sent_idx])
        # ==============================================================================
        assert doc_start_end_pair_list[-1][1] + 1 == concat_len
        assert len(ctx_token2sent_map_list) == concat_len, '{}, {}'.format(len(ctx_token2sent_map_list), concat_len)
        assert len(doc_labels) == len(doc_start_end_pair_list)
        assert previous_len == len(concat_encode) and previous_len == concat_len
        assert len(ctx_sent2doc_map_list) == len(supp_sent_labels_list)
        assert len(ctx_sentIndoc_idx_list) == len(supp_sent_labels_list)
        assert len(supp_sent_labels_list) == len(sent_start_end_pair_list) and concat_sent_num == len(
            sent_start_end_pair_list)

        doc_start_end_pair_list, ctx_lens = position_filter(doc_start_end_pair_list, max_len=self.max_token_num)
        sent_start_end_pair_list, ctx_sent_lens = position_filter(sent_start_end_pair_list, max_len=self.max_token_num)
        if doc_num < self.max_doc_num:
            pad_doc_num = self.max_doc_num - doc_num
            doc_start_end_pair_list = doc_start_end_pair_list + [(0, 0)] * pad_doc_num
            ctx_lens = ctx_lens + [0] * pad_doc_num
            doc_labels = doc_labels + [0] * pad_doc_num
            doc_sent_nums = doc_sent_nums + [0] * pad_doc_num
            # ================================================
            ctx_title_lens = ctx_title_lens + [0] * pad_doc_num
            # ================================================
        ###############################################################################################################
        ss_attn_mask, sd_attn_mask = mask_generation(sent_num_docs=doc_sent_nums, max_sent_num=self.max_sent_num)
        ###############################################################################################################
        if concat_sent_num < self.max_sent_num:
            pad_sent_num = self.max_sent_num - concat_sent_num
            sent_start_end_pair_list = sent_start_end_pair_list + [(0, 0)] * pad_sent_num
            ctx_sent_lens = ctx_sent_lens + [0] * pad_sent_num
            supp_sent_labels_list = supp_sent_labels_list + [0] * pad_sent_num
            #########+++++++++++++++++++++++++++++++++++
            ctx_sent2doc_map_list = ctx_sent2doc_map_list + [0] * pad_sent_num
            ctx_sentIndoc_idx_list = ctx_sentIndoc_idx_list + [0] * pad_sent_num
            #########+++++++++++++++++++++++++++++++++++
        if not yes_no_question:
            answer_position = answer_position_list[0]
        else:
            answer_position = (0, 0)

        if len(ctx_token2sent_map_list) < self.max_token_num:
            pad_token_num = self.max_token_num - len(ctx_token2sent_map_list)
            ctx_token2sent_map_list = ctx_token2sent_map_list + [0] * pad_token_num

        cat_doc_encodes = self.hotpot_tensorizer.token_ids_to_tensor(token_ids=concat_encode)
        cat_doc_attention_mask = self.hotpot_tensorizer.get_attn_mask(token_ids_tensor=cat_doc_encodes)
        if self.global_mask_type == 'query':
            query_mask_idxes = [x for x in range(query_len)]
        elif self.global_mask_type == 'query_doc':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list]
        elif self.global_mask_type == 'query_doc_sent':
            query_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[0] for x in
                                                                                                          sent_start_end_pair_list]
        else:
            query_mask_idxes = [x for x in range(query_len)]
        cat_doc_global_attn_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes,
                                                                               gobal_mask_idxs=query_mask_idxes)
        #######
        marker_mask_idxes = [x for x in range(query_len)] + [x[0] for x in doc_start_end_pair_list] + [x[0] for x in
                                                                                                       sent_start_end_pair_list]
        # +++++++++++++++++++++++++++++++++++++++++++++++++
        title_mask_idxes = []
        for t_idx, ti_len in enumerate(ctx_title_lens):
            if ti_len > 0:
                title_start_idx = doc_start_end_pair_list[t_idx][0] + 1
                for j in range(ti_len):
                    title_mask_idxes.append(title_start_idx + j)
        marker_mask_idxes = marker_mask_idxes + title_mask_idxes
        # +++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_marker_mask = self.hotpot_tensorizer.get_global_attn_mask(tokens_ids_tensor=cat_doc_encodes,
                                                                      gobal_mask_idxs=marker_mask_idxes)
        ctx_marker_mask = ctx_marker_mask.type(torch.bool)
        #######
        doc_start_idxes = torch.LongTensor([x[0] for x in doc_start_end_pair_list])
        doc_end_idxes = torch.LongTensor([x[1] for x in doc_start_end_pair_list])
        sent_start_idxes = torch.LongTensor([x[0] for x in sent_start_end_pair_list])
        sent_end_idxes = torch.LongTensor([x[1] for x in sent_start_end_pair_list])
        answer_start_idx, answer_end_idx = torch.LongTensor([answer_position[0]]), torch.LongTensor([answer_position[1]])
        doc_lens = torch.LongTensor(ctx_lens)
        doc_labels = torch.LongTensor(doc_labels)
        ctx_sent_lens = torch.LongTensor(ctx_sent_lens)
        supp_sent_labels = torch.LongTensor(supp_sent_labels_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_sent2doc_map = torch.LongTensor(ctx_sent2doc_map_list)
        ctx_sentIndoc_idx = torch.LongTensor(ctx_sentIndoc_idx_list)
        ctx_token2sent_map = torch.LongTensor(ctx_token2sent_map_list)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert concat_len <= self.max_token_num
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return cat_doc_encodes, cat_doc_attention_mask, cat_doc_global_attn_mask, doc_start_idxes, sent_start_idxes, \
               answer_start_idx, answer_end_idx, doc_lens, doc_labels, ctx_sent_lens, supp_sent_labels, yes_no_label, \
               ss_attn_mask, sd_attn_mask, ctx_sent2doc_map, ctx_sentIndoc_idx, ctx_token2sent_map, ctx_marker_mask, \
               doc_end_idxes, sent_end_idxes, concat_len, concat_sent_num

    @staticmethod
    def collate_fn(data):
        batch_max_ctx_len = max([_[20] for _ in data])
        batch_max_sent_num = max([_[21] for _ in data])
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ctx_sample = torch.stack([_[0] for _ in data], dim=0)
        batch_ctx_mask_sample = torch.stack([_[1] for _ in data], dim=0)
        batch_ctx_global_sample = torch.stack([_[2] for _ in data], dim=0)

        batch_ctx_sample = batch_ctx_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_mask_sample = batch_ctx_mask_sample[:, range(0, batch_max_ctx_len)]
        batch_ctx_global_sample = batch_ctx_global_sample[:, range(0, batch_max_ctx_len)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_starts = torch.stack([_[3] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_starts = torch.stack([_[4] for _ in data], dim=0)
        batch_sent_starts = batch_sent_starts[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_answer_starts = torch.stack([_[5] for _ in data], dim=0)
        batch_answer_ends = torch.stack([_[6] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_lens = torch.stack([_[7] for _ in data], dim=0)
        batch_doc_labels = torch.stack([_[8] for _ in data], dim=0)
        batch_sent_lens = torch.stack([_[9] for _ in data], dim=0)
        batch_sent_lens = batch_sent_lens[:, range(0, batch_max_sent_num)]
        batch_sent_labels = torch.stack([_[10] for _ in data], dim=0)
        batch_sent_labels = batch_sent_labels[:, range(0, batch_max_sent_num)]
        batch_yes_no = torch.stack([_[11] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_ss_attn_mask = torch.stack([_[12] for _ in data], dim=0)
        batch_sd_attn_mask = torch.stack([_[13] for _ in data], dim=0)
        batch_ss_attn_mask = batch_ss_attn_mask[:, range(0, batch_max_sent_num)][:, :, range(0, batch_max_sent_num)]
        batch_sd_attn_mask = batch_sd_attn_mask[:, :, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent2doc_map = torch.stack([_[14] for _ in data], dim=0)
        batch_sent2doc_map = batch_sent2doc_map[:, range(0, batch_max_sent_num)]
        batch_sentIndoc_map = torch.stack([_[15] for _ in data], dim=0)
        batch_sentIndoc_map = batch_sentIndoc_map[:, range(0, batch_max_sent_num)]
        batch_token2sent = torch.stack([_[16] for _ in data], dim=0)
        batch_token2sent = batch_token2sent[:, range(0, batch_max_ctx_len)]
        batch_marker = torch.stack([_[17] for _ in data], dim=0)
        batch_marker = batch_marker[:, range(0, batch_max_ctx_len)]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_doc_ends = torch.stack([_[18] for _ in data], dim=0)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        batch_sent_ends = torch.stack([_[19] for _ in data], dim=0)
        batch_sent_ends = batch_sent_ends[:, range(0, batch_max_sent_num)]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {'ctx_encode': batch_ctx_sample, 'ctx_attn_mask': batch_ctx_mask_sample,
               'ctx_global_mask': batch_ctx_global_sample, 'doc_start': batch_doc_starts, 'doc_end': batch_doc_ends,
                'sent_start': batch_sent_starts, 'sent_end': batch_sent_ends, 'ans_start': batch_answer_starts, 'ans_end': batch_answer_ends,
               'doc_lens': batch_doc_lens, 'doc_labels': batch_doc_labels, 'sent_lens': batch_sent_lens,
               'sent_labels': batch_sent_labels, 'yes_no': batch_yes_no, 'ss_mask': batch_ss_attn_mask, 'sd_mask': batch_sd_attn_mask,
               's2d_map': batch_sent2doc_map, 'sInd_map': batch_sentIndoc_map, 'marker': batch_marker,
               't2s_map': batch_token2sent}
        return res
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##**********************************************************************************************************************
##**********************************************************************************************************************