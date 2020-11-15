import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import load_model
import argparse
import logging
import os
from time import time
import torch
import pandas as pd
from torch import Tensor as T
from torch.nn import DataParallel
from modelTrain.QATrainFunction import get_date_time, read_train_dev_data_frame
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE, get_hotpotqa_longformer_tokenizer

from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from torch.utils.data import DataLoader
from multihopQA.hotpotQAdataloader import HotpotTestDataset

def get_model(args):
    start_time = time()
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name)
    longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                 hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                 seq_project=args.seq_project)
    longEncoder.resize_token_embeddings(len(tokenizer))
    model = LongformerHotPotQAModel(longformer=longEncoder, num_labels=args.num_labels, args=args)
    logging.info('Constructing reasonModel completes in {:.4f}'.format(time() - start_time))
    return model

def get_test_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.test_data_name)
    batch_size = args.test_batch_size
    data_size = data_frame.shape[0]
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotTestDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotTestDataset.collate_fn
    )
    return dataloader, data_size

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Retrieval Models',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_test', default=True, action='store_true')
    parser.add_argument('--model_path', type=str, default='../model')
    parser.add_argument('--model_name', type=str, default='../model')
    parser.add_argument('--test_data_name', type=str, default='hotpot_dev_distractor_wiki_tokenized.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--test_log_steps', default=10, type=int, help='valid/test log every xx steps')
    parser.add_argument('--save_path', type=str, default='../test_result')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/distractor_qa')
    parser.add_argument('--dev_data_name', type=str, default='hotpot_dev_distractor_wiki_tokenized.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--orig_data_path', type=str, default='../data/hotpotqa')
    parser.add_argument('--orig_dev_data_name', type=str, default='hotpot_dev_distractor_v1.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    date_time_str = get_date_time()
    log_file = os.path.join(args.save_path, date_time_str + '_test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main(args):
    if args.data_path is None:
        raise ValueError('one of data_path must be chosed.')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    ########+++++++++++++++++++++++++++++
    abs_path = os.path.abspath(args.data_path)
    args.data_path = abs_path
    ########+++++++++++++++++++++++++++++
    # Write logs to checkpoint and console

    if args.cuda:
        device = torch.device("cuda:0")
        logging.info('GPU setting')
    else:
        device = torch.device('cpu')
        logging.info('CPU setting')
    ########+++++++++++++++++++++++++++++
    logging.info('Loading training data...')
    logging.info('Loading development data...')
    test_data_loader, _ = get_test_data_loader(args=args)
    logging.info('Loading data completed')
    logging.info('*'*75)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.do_test:
        logging.info('Loading reasonModel...')
        model = get_model(args=args).to(device)
        ##+++++++++++
        model_path = args.save_path
        model_file_name = args.check_point
        hotpot_qa_model_name = os.path.join(model_path, model_file_name)
        model, _, _, _= load_model(model=model, PATH=hotpot_qa_model_name)
        model = model.to(device)
        ##+++++++++++
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
        logging.info('*' * 75)
        logging.info("Model hype-parameter information...")
        for key, value in vars(args).items():
            logging.info('Hype-parameter\t{} = {}'.format(key, value))
        logging.info('*' * 75)
        logging.info('projection_dim = {}'.format(args.project_dim))
        logging.info('Testing on dataset...')
        logging.info('*' * 75)
        test_result_data = hotpot_prediction(model=model, test_data_loader=test_data_loader, args=args)
        test_result_name = os.path.join(args.save_path, 'prediction.json')
        test_result_data.to_json(test_result_name, orient='records')
        logging.info('Saving {} record results to {}'.format(test_result_data.shape, test_result_name))
        logging.info('*' * 75)

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hotpot_prediction(model, test_data_loader, args):
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

if __name__ == '__main__':
    main(parse_args())