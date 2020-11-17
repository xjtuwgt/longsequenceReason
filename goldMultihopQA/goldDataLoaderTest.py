import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from time import time
from multihopUtils.longformerQAUtils import LongformerQATensorizer, get_hotpotqa_longformer_tokenizer
from multihopUtils.hotpotqaIOUtils import loadWikiData as read_train_dev_data_frame
from goldMultihopQA.goldHotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset

def data_loader_consistent_checker(train=True):
    file_path = '../data/hotpotqa/distractor_qa'
    if train:
        dev_file_name = 'hotpot_train_distractor_wiki_tokenized.json'
    else:
        dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
    longtokenizer = get_hotpotqa_longformer_tokenizer()
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    from torch.utils.data import DataLoader
    batch_size = 1
    if train:
        dev_dataloader = DataLoader(
            HotpotTrainDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=HotpotTrainDataset.collate_fn
        )
    else:
        dev_dataloader = DataLoader(
            HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=HotpotDevDataset.collate_fn
        )

    head_two = data_frame.head(batch_size)
    print(type(head_two))
    for idx, row in head_two.iterrows():
        context = row['context']
        supp_fact_filtered = row['supp_facts_filtered']
        for supp, sen_idx in supp_fact_filtered:
            print('Support doc: {}, sent id: {}'.format(supp, sen_idx))
            print('-' * 70)
        print()
        print('Query {}'.format(row['question']))
        for doc_idx, doc in enumerate(context):
            print('doc {}: title = {} \n text = {}'.format(doc_idx + 1, doc[0], ' '.join(doc[1])))
            print('-' * 70)
        print('*' * 70)
        print()
        print('Original answer = {}'.format(row['norm_answer']))
        print('=' * 70)
    print('+' * 70)
    print('\n'*5)
    for batch_idx, sample in enumerate(dev_dataloader):
        ctx_encode = sample['ctx_encode']
        doc_start = sample['doc_start'].squeeze(dim=-1)
        sent_start = sample['sent_start'].squeeze(dim=-1)
        answer_start = sample['ans_start'].squeeze(dim=-1)
        answer_end = sample['ans_end'].squeeze(dim=-1)
        if train:
            head_idx = sample['head_idx'].squeeze(dim=-1)
            tail_idx = sample['tail_idx'].squeeze(dim=-1)
        sent_lens = sample['sent_lens'].squeeze(dim=-1)
        attention = sample['ctx_attn_mask'].squeeze(dim=-1)
        global_attenion = sample['ctx_global_mask']
        print('global attention {}'.format(global_attenion))
        marker = sample['marker'].squeeze(dim=-1)

        doc_num = doc_start.shape[1]
        print('doc num: {}'.format(doc_start.shape))
        print('marker {}'.format(marker))
        print('marker shape {}'.format(marker.shape))

        for idx in range(ctx_encode.shape[0]):
            ctx_i = ctx_encode[idx]
            marker_i = marker[idx]

            marker_idx = marker_i.nonzero().squeeze()
            print('marker text {}'.format(longtokenizer.decode(ctx_i[marker_idx])))
            print('*' * 75)
            attention_i = attention[idx]
            attn_idx = (attention_i == 1).nonzero().squeeze()
            print('attn text {}'.format(longtokenizer.decode(ctx_i[attn_idx])))
            sent_start_i = sent_start[idx]
            doc_start_i = doc_start[idx]
            if train:
                head_i = head_idx[idx].data.item()
                tail_i = tail_idx[idx].data.item()
            ans_start_i = answer_start[idx].data.item()
            ans_end_i = answer_end[idx].data.item()

            print('Decode Query {}'.format(longtokenizer.decode(ctx_i[:doc_start_i[0]])))
            print('*' * 75)
            print('Decoded answer = {}'.format(hotpot_tensorizer.to_string(ctx_i[ans_start_i:(ans_end_i + 1)])))
            print('*' * 75)
            # print(ans_start_i)

            doc_marker = longtokenizer.decode(ctx_i[doc_start_i])
            print('doc_marker: {}'.format(doc_marker))

            sent_marker = longtokenizer.decode(ctx_i[sent_start_i])
            print('doc: {}\nsent: {}\n{}\n{}'.format(doc_marker, sent_marker, sent_start_i.shape, sent_lens[idx]))
            print('*' * 75)


            for k in range(doc_num):
                if k < doc_num - 1:
                    # doc_k = hotpot_tensorizer.to_string(ctx_i[doc_start_i[k]:doc_start_i[k+1]])
                    doc_k = longtokenizer.decode(ctx_i[doc_start_i[k]:doc_start_i[k+1]])
                else:
                    # doc_k = hotpot_tensorizer.to_string(ctx_i[doc_start_i[k]:])
                    doc_k = longtokenizer.decode(ctx_i[doc_start_i[k]:])
                # print(doc_marker)
                print('Supp doc {}: text = {}'.format(k+1, doc_k))
                if train:
                    if k == head_i:
                        print('=' * 70)
                        print('Head positive doc {}: text: {}'.format(head_i + 1, doc_k))
                        print('=' * 70)
                    if k == tail_i:
                        print('=' * 70)
                        print('Tail positive doc {}: text: {}'.format(tail_i + 1, doc_k))
                        print('=' * 70)
                    print('-'*70)
            print('*' * 70)
            print()
        # print(ctx_encode.shape)
        break
    print('Runtime = {}'.format(time() - start_time))


def data_loader_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    batch_size = 6

    data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
    for col in data_frame.columns:
        print(col)
    longtokenizer = get_hotpotqa_longformer_tokenizer()
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    dev_dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=HotpotDevDataset.collate_fn
    )

    for batch_idx, sample in enumerate(dev_dataloader):
        x= sample['doc_start']
        # print(sample['doc_start'].shape)
        # print(sample['sent_start'].shape)
    print('Runtime = {}'.format(time() - start_time))

def test_data_loader_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    batch_size = 1
    data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
    longtokenizer = get_hotpotqa_longformer_tokenizer()
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    test_dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=HotpotDevDataset.collate_fn
    )
    for batch_idx, sample in enumerate(test_dataloader):
        sd_mask = sample['sd_mask']
        # print(sd_mask)
        # print(sd_mask[0])
        print(sample['doc_lens'])
        print(sample['sent_lens'])

        ss_mask = sample['ss_mask']
        # print(ss_mask[0].detach().tolist())
        print(ss_mask.shape)
        print(ss_mask[0].sum(dim=1))
        print(sd_mask.shape)
        break
    print('Runtime = {}'.format(time() - start_time))

def data_consistent_checker(train=True):
    file_path = '../data/hotpotqa/distractor_qa'
    from torch.utils.data import DataLoader
    batch_size = 2
    longtokenizer = get_hotpotqa_longformer_tokenizer()
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    if train:
        dev_file_name = 'hotpot_train_distractor_wiki_tokenized.json'
        data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
        start_time = time()
        dev_dataloader = DataLoader(
            HotpotTrainDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=HotpotTrainDataset.collate_fn
        )
    else:
        dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
        data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
        start_time = time()
        dev_dataloader = DataLoader(
            HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=HotpotDevDataset.collate_fn
        )

    batch_data_frame = data_frame.head(batch_size)
    print(batch_data_frame.shape)
    for idx, row in batch_data_frame.iterrows():
        context = row['context']
        supp_fact_filtered = row['supp_facts_filtered']
        # for supp, sen_idx in supp_fact_filtered:
        #     print('Support doc: {}, sent id: {}'.format(supp, sen_idx))
        print('Query {}'.format(row['question']))
        for doc_idx, doc in enumerate(context):
            # print('doc {}: title = {} \n text = {}'.format(doc_idx + 1, doc[0], '\n'.join(doc[1])))
            print('doc {}: title = {}'.format(doc_idx + 1, doc[0]))
            for supp, sen_idx in supp_fact_filtered:
                if doc[0] == supp:
                    print('supp fact doc {}: sent = {} text = {}'.format(doc_idx, sen_idx, doc[1][sen_idx]))
        print('*' * 70)
        print('Original answer = {}'.format(row['norm_answer']))
        print('=' * 70)
    print('+' * 70)
    print('\n' *3)

    for batch_idx, sample in enumerate(dev_dataloader):
        # for key, value in sample.items():
        #     print(key)
        ctx_encode = sample['ctx_encode']
        ctx_marker_mask = sample['marker']
        global_atten = sample['ctx_global_mask']
        atten_mask = sample['ctx_attn_mask']
        sup_sent_labels = sample['sent_labels'].squeeze(dim=-1)
        sent2doc_map = sample['s2d_map']
        sentIndoc_map = sample['sInd_map']
        sent_start = sample['sent_start']
        sent_end = sample['sent_end']
        # print('sent num = {}'.format(sent_end.shape[1]))
        answer_start = sample['ans_start'].squeeze(dim=-1)
        answer_end = sample['ans_end'].squeeze(dim=-1)
        doc_start = sample['doc_start'].squeeze(dim=-1)
        token2sent_map = sample['t2s_map'].squeeze(dim=-1)
        if train:
            head_idx = sample['head_idx'].squeeze(dim=-1)
            tail_idx = sample['tail_idx'].squeeze(dim=-1)

        for id in range(batch_size):
            ctx_marker_i = ctx_marker_mask[id]
            supp_idxes = (sup_sent_labels[id] > 0).nonzero().squeeze()
            doc_idxes = sent2doc_map[id][supp_idxes].detach().tolist()
            sent_idxes = sentIndoc_map[id][supp_idxes].detach().tolist()
            doc_start_i = doc_start[id]
            doc_sent_pairs = list(zip(doc_idxes, sent_idxes))
            sent_start_i = sent_start[id]
            sent_end_i = sent_end[id]
            ctx_encode_i = ctx_encode[id]
            token2sent_map_i = token2sent_map[id]

            # print('token to sentence {}'.format(token2sent_map_i.max()))
            max_sent_num = token2sent_map_i.max().data.item()
            for ssss_id in range(max_sent_num):
                sent_iiii_idexs = (token2sent_map_i == ssss_id).nonzero().squeeze()
                print('sent {} text = {}'.format(ssss_id, longtokenizer.decode(ctx_encode_i[sent_iiii_idexs])))

            if train:
                print('head doc idx = {}'.format(head_idx[id]))
                print('tail doc idx = {}'.format(tail_idx[id]))

            global_atten_i = global_atten[id]
            global_atten_i_indexes = (global_atten_i > 0).nonzero().squeeze()
            global_atten_text = longtokenizer.decode(ctx_encode_i[global_atten_i_indexes])
            print('global attention text: {}'.format(global_atten_text))

            atten_i = atten_mask[id]
            atten_i_indexes = (atten_i > 0).nonzero().squeeze()
            atten_text = longtokenizer.decode(ctx_encode_i[atten_i_indexes])
            # print('attention text: {}'.format(atten_text))
            print('x'*75)
            # print('decode text: {}'.format(longtokenizer.decode(ctx_encode_i)))

            ans_start_i = answer_start[id].data.item()
            ans_end_i = answer_end[id].data.item()
            #
            print('Decode Query {}'.format(longtokenizer.decode(ctx_encode_i[:doc_start_i[0]])))
            print('Decode Answer {}'.format(longtokenizer.decode(ctx_encode_i[ans_start_i:(ans_end_i +1)])))

            ctx_marker_i_indexes = (ctx_marker_i > 0).nonzero().squeeze()
            print('Decode marker text = {}'.format(longtokenizer.decode(ctx_encode_i[ctx_marker_i_indexes])))
            for ss_id, x in enumerate(doc_sent_pairs):
                supp_idddd = supp_idxes[ss_id]
                start_i, end_i = sent_start_i[supp_idddd], sent_end_i[supp_idddd] + 1
                print('doc {}, sent {}, text {}'.format(x[0], x[1], longtokenizer.decode(ctx_encode_i[start_i:end_i])))
            print('=' * 70)
        break
    return

def answer_consistent_checker():
    file_path = '../data/hotpotqa/distractor_qa'
    dev_file_name = 'hotpot_dev_distractor_wiki_tokenized.json'
    from torch.utils.data import DataLoader
    batch_size = 1

    data_frame = read_train_dev_data_frame(PATH=file_path, json_fileName=dev_file_name)
    print(data_frame['answer_len'].max())
    # for col in data_frame.columns:
    #     print(col)
    longtokenizer = get_hotpotqa_longformer_tokenizer()
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=longtokenizer, max_length=4096)
    start_time = time()
    dev_dataloader = DataLoader(
        HotpotTrainDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=14,
        collate_fn=HotpotTrainDataset.collate_fn
    )
    max_seq_len = 0
    average_seq_len = 0
    count = 0
    max_answer_len = 0
    for batch_idx, sample in enumerate(dev_dataloader):
        # if batch_idx % 1000 == 0:
        #     print(batch_idx)
        ctx_encode = sample['ctx_encode']
        ctx_encode_lens = sample['doc_lens']


        answer_start = sample['ans_start'].squeeze(dim=-1)
        answer_end = sample['ans_end'].squeeze(dim=-1)
        doc_start = sample['doc_start'].squeeze(dim=-1)
        doc_end = sample['doc_end'].squeeze(dim=-1)
        sent_start = sample['sent_start'].squeeze(dim=-1)
        batch_size = ctx_encode.shape[0]
        for id in range(batch_size):
            # doc_token_num = ctx_encode_lens[id].sum().data.item()
            doc_token_num = doc_end[id].detach().tolist()[-1]
            if max_seq_len < doc_token_num:
                max_seq_len = doc_token_num
            average_seq_len = average_seq_len + doc_token_num
            count = count + 1
            doc_start_i = doc_start[id]
            sent_start_i = sent_start[id]
            ctx_encode_i = ctx_encode[id]
            ans_start_i = answer_start[id].data.item()
            ans_end_i = answer_end[id].data.item()
            if max_answer_len < (ans_end_i - ans_start_i) + 1:
                max_answer_len = (ans_end_i - ans_start_i) + 1
            decode_answer = longtokenizer.decode(ctx_encode_i[ans_start_i:(ans_end_i +1)])

            print('{}\t{}\t{}'.format(batch_idx, decode_answer, ctx_encode.shape))
            # if '<p>' in decode_answer or '<d>' in decode_answer or '<q>' in decode_answer or '</q>' in decode_answer:
            #     print('index = {}'.format(batch_idx))
            #     print('decode answer {}'.format(decode_answer))
            #     print('Decode Query {}'.format(longtokenizer.decode(ctx_encode_i[:doc_start_i[0]])))
            # print('decode answer {}'.format(decode_answer))

    print('max seq len: {} average seq len: {}, {}'.format(max_seq_len, average_seq_len/count, count))
    print('max answer len: {}'.format(max_answer_len))
    return

if __name__ == '__main__':
    data_loader_consistent_checker(False)
    # data_loader_checker()
    # test_data_loader_checker()
    # data_consistent_checker(train=True)
    # answer_consistent_checker()
    # data, _ = HOTPOT_DevData_Distractor()
    # for r_idx, row in data.iterrows():
    #     print('{}\t{}'.format(r_idx, row['answer']))
    print()