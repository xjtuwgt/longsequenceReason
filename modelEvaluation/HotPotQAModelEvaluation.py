import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import load_model
import logging
import argparse
import os
from time import time
import torch
import json
from torch.nn import DataParallel
from multihopUtils.gpu_utils import gpu_setting
from modelTrain.QATrainFunction import get_date_time, read_train_dev_data_frame, test_all_steps, log_metrics
from multihopUtils.longformerQAUtils import get_hotpotqa_longformer_tokenizer

from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder
from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
from torch.utils.data import DataLoader
from multihopQA.hotpotQAdataloader import HotpotDevDataset
from modelEvaluation.hierarchicalDecoder import test_all_steps_hierartical

######
MODEL_PATH = '../model'
######

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Testing Long Sequence Reason Model')
    parser.add_argument('--model_name', default='60000_0.007160362892318517_0.8174847929473423.pt', help='use GPU')
    parser.add_argument('--model_config_name', default='config.json', help='use GPU')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/distractor_qa')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--orig_data_path', type=str, default='../data/hotpotqa')
    parser.add_argument('--orig_dev_data_name', type=str, default='hotpot_dev_distractor_v1.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--dev_data_name', type=str, default='hotpot_dev_distractor_wiki_tokenized.json')
    parser.add_argument('--test_batch_size', type=int, default=54)
    parser.add_argument('--doc_topk', type=int, default=2)
    parser.add_argument('--doc_threshold', type=float, default=0.6)
    parser.add_argument('--sent_threshold', type=float, default=0.8)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return parser.parse_args(args)

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

def get_config(PATH, config_json_name):
    parser = argparse.ArgumentParser(
        description='Training and Testing Long Sequence Reason Model',
        usage='train.py [<args>] [-h | --help]')
    config_json_file = os.path.join(PATH, config_json_name)
    with open(config_json_file, 'r') as config_file:
        config_data = json.load(config_file)

    for key, value in config_data.items():
        parser.add_argument('--' + key, default=value)
    return parser.parse_args()


def get_test_data_loader(args):
    data_frame = read_train_dev_data_frame(file_path=args.data_path, json_fileName=args.dev_data_name)
    batch_size = args.test_batch_size
    data_size = data_frame.shape[0]
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name, do_lower_case=True)
    hotpot_tensorizer = LongformerQATensorizer(tokenizer=tokenizer, max_length=args.max_ctx_len)
    dataloader = DataLoader(
        HotpotDevDataset(data_frame=data_frame, hotpot_tensorizer=hotpot_tensorizer, max_sent_num=args.max_sent_num,
                      global_mask_type=args.global_mask_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDevDataset.collate_fn
    )
    return dataloader, data_size

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

def main(model_args):
    args = get_config(PATH=MODEL_PATH, config_json_name=model_args.model_config_name)
    args.check_point = model_args.model_name
    args.data_path = model_args.data_path
    args.test_batch_size = model_args.test_batch_size
    args.doc_threshold = model_args.doc_threshold
    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    ###################
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
        if args.gpu_num > 1:
            device_ids, used_memory = gpu_setting(args.gpu_num)
        else:
            device_ids, used_memory = gpu_setting()
        if used_memory > 100:
            logging.info('Using memory = {}'.format(used_memory))
        if device_ids is not None:
            if len(device_ids) > args.gpu_num:
                device_ids = device_ids[:args.gpu_num]
            device = torch.device('cuda:{}'.format(device_ids[0]))
        else:
            device = torch.device('cuda:0')
        logging.info('Set the cuda with idxes = {}'.format(device_ids))
        logging.info('cuda setting {}'.format(device))
        logging.info('GPU setting')
    else:
        device_ids = None
        device = torch.device('cpu')
        logging.info('CPU setting')
    ########+++++++++++++++++++++++++++++
    logging.info('Loading development data...')
    test_data_loader, _ = get_test_data_loader(args=args)
    logging.info('Loading data completed')
    logging.info('*'*75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Loading Model...')
    model = get_model(args=args).to(device)
    ##+++++++++++
    model_path = args.save_path
    model_file_name = args.check_point
    hotpot_qa_model_name = os.path.join(model_path, model_file_name)
    model = load_model(model=model, PATH=hotpot_qa_model_name)
    model = model.to(device)
    if device_ids is not None:
        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids, output_device=device)
            logging.info('Data Parallel model setting')
    ##+++++++++++
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logging.info('*' * 75)
    logging.info("Model hype-parameter information...")
    for key, value in vars(args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    logging.info("Model hype-parameter information...")
    for key, value in vars(model_args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    logging.info('projection_dim = {}'.format(args.project_dim))
    logging.info('Multi-task encoding')
    logging.info('*' * 75)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    # metric_dict = test_all_steps(model=model, device=device, test_data_loader=test_data_loader, args=args)
    # answer_type_acc = metric_dict['answer_type_acc']
    # logging.info('*' * 75)
    # logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
    # logging.info('*' * 75)
    # for key, value in metric_dict.items():
    #     if key.endswith('metrics'):
    #         logging.info('{} prediction'.format(key))
    #         log_metrics('Valid', 'final', value)
    # logging.info('*' * 75)
    # ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    # dev_data_frame = metric_dict['res_dataframe']
    # date_time_str = get_date_time()
    # dev_result_name = os.path.join(args.save_path,
    #                                date_time_str + '_' + args.check_point + '_evaluation.json')
    # dev_data_frame.to_json(dev_result_name, orient='records')
    # logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
    # logging.info('*' * 75)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Hierarchical encoding')
    metric_dict = test_all_steps_hierartical(model=model, device=device, test_data_loader=test_data_loader, doc_topk=model_args.doc_topk, args=args)
    answer_type_acc = metric_dict['answer_type_acc']
    logging.info('*' * 75)
    logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metrics'):
            logging.info('{} prediction'.format(key))
            log_metrics('Valid', 'final', value)
        logging.info('*' * 75)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++
    dev_data_frame = metric_dict['res_dataframe']
    date_time_str = get_date_time()
    dev_result_name = os.path.join(args.save_path,
                                   date_time_str + '_' + args.check_point + '_hier_evaluation.json')
    dev_data_frame.to_json(dev_result_name, orient='records')
    logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
    logging.info('*' * 75)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    main(parse_args())
    print()