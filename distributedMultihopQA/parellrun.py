import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import argparse
import logging
import os
import torch
from multihopUtils.gpu_utils import gpu_setting
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE
from distributedMultihopQA.parQATrainFunction import run_train_and_dev


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Retrieval Models',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_debug', action='store_true', help='whether')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_retrieval', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/distractor_qa')
    parser.add_argument('--train_data_name', type=str, default='hotpot_train_distractor_wiki_tokenized.json')
    parser.add_argument('--train_data_filtered', type=int, default=0)
    parser.add_argument('--dev_data_name', type=str, default='hotpot_dev_distractor_wiki_tokenized.json')
    parser.add_argument('--model', default='Unified QA Model', type=str)
    parser.add_argument('--pretrained_cfg_name', default=PRE_TAINED_LONFORMER_BASE, type=str)
    parser.add_argument('--gpu_num', default=4, type=int)
    parser.add_argument('--world_size', default=4, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('--gamma', default=2.0, type=float, help='parameter for focal loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='parameter for focal loss')
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--model_name', default='MLP', type=str) #'DotProduct', 'BiLinear', 'MLP'
    parser.add_argument('--hop_model_name', default='BiLinear', type=str)  # 'DotProduct', 'BiLinear'
    parser.add_argument('--hop_score', default=False, type=bool)
    parser.add_argument('--frozen_layer_num', default=2, type=int, help='number of layers for document encoder frozen during training')
    parser.add_argument('--pad_neg_samp_size', default=8, type=int)
    parser.add_argument('--project_dim', default=256, type=int)
    parser.add_argument('--global_mask_type', default='query_doc', type=str) ## query, query_doc, query_doc_sent
    parser.add_argument('--training_shuffle', default=0, type=int)  ## whether re-order training data
    parser.add_argument('--sent_threshold', default=0.925, type=float)
    parser.add_argument('--max_sent_num', default=150, type=int)
    parser.add_argument('--max_ctx_len', default=4096, type=int)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.00004, type=float) # 1e-5 level
    parser.add_argument('--batch_size', default=2, type=int)
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##+++++++++++++++++++++++++Parameter for multi-head self-attention++++++++++++++++++++++++++++++++++
    parser.add_argument('--input_drop', default=0.1, type=float)
    parser.add_argument('--attn_drop', default=0.1, type=float)
    parser.add_argument('--heads', default=8, type=float)
    parser.add_argument('--with_graph', default=True, type=bool)
    parser.add_argument('--span_weight', default=0.1, type=float)
    parser.add_argument('--pair_score_weight', default=0.1, type=float)
    parser.add_argument('--seq_project', default=True, action='store_true', help='whether perform sequence projection')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--num_labels', default=2, type=int,
                        help='span prediction label')  ##start and end position prediction, seperately
    parser.add_argument('--grad_clip_value', default=10.0, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='../model', type=str)
    parser.add_argument('--max_steps', default=60000, type=int)
    parser.add_argument('--epoch', default=6, type=int)
    parser.add_argument('--warm_up_steps', default=2000, type=int)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=50, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=10, type=int, help='valid/test log every xx steps')
    parser.add_argument('--rand_seed', default=12345, type=int, help='random seed')

    return parser.parse_args(args)

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

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
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ########+++++++++++++++++++++++++++++
    abs_path = os.path.abspath(args.data_path)
    args.data_path = abs_path
    ########+++++++++++++++++++++++++++++
    # Write logs to checkpoint and console
    set_logger(args)

    if args.do_debug:
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
        for id in device_ids:
            x = torch.device("cuda:%d" % id)
        logging.info('Set the cuda with idxes = {}'.format(device_ids))
        logging.info('cuda setting {}'.format(device))
    else:
        if args.gpu_num > 1:
            logging.info("Using GPU!")
            available_device_count = torch.cuda.device_count()
            # ++++++++++++++++++++++++++++++++++
            gpu_setting(available_device_count)
            # ++++++++++++++++++++++++++++++++++
            logging.info('GPU number is {}'.format(available_device_count))
            if args.gpu_num > available_device_count:
                args.gpu_num = available_device_count
            device_ids = list(range(args.gpu_num))
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:0")
            device_ids = [0]
            logging.info('Single GPU setting')
        logging.info('Set the cuda with idxes = {}'.format(device_ids))
        logging.info('cuda setting {}'.format(device))
        args.gpu_num = len(device_ids)
        args.world_size = len(device_ids)

    logging.info('Available gpu number = {}, world_size = {}'.format(args.gpu_num, args.world_size))

    args.world_size = 2
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('*' * 75)
    logging.info("Model hype-parameter information...")
    for key, value in vars(args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    logging.info('batch_size = {}'.format(args.batch_size))
    logging.info('projection_dim = {}'.format(args.project_dim))
    logging.info('learning_rate = {}'.format(args.learning_rate))
    logging.info('Start training...')
    if args.do_train:
        logging.info('Start model training...')
        run_train_and_dev(args=args)

if __name__ == '__main__':
    main(parse_args())