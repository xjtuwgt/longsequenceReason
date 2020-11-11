import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import save_check_point
import argparse
import logging
import os
from time import time
import torch
from torch.nn import DataParallel
from multihopUtils.gpu_utils import gpu_setting, set_seeds
from modelTrain.QATrainFunction import get_train_data_loader, get_dev_data_loader, get_model, get_date_time, get_check_point
from modelTrain.QATrainFunction import train_all_steps, test_all_steps, log_metrics
from modelTrain.EfficientQATraining import train_all_steps as efficient_train_all_steps
from multihopUtils.longformerQAUtils import PRE_TAINED_LONFORMER_BASE

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Long Sequence Reason Model',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_debug', action='store_true', help='whether')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/distractor_qa')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--orig_data_path', type=str, default='../data/hotpotqa')
    parser.add_argument('--orig_dev_data_name', type=str, default='hotpot_dev_distractor_v1.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--train_data_name', type=str, default='hotpot_train_distractor_wiki_tokenized.json')
    parser.add_argument('--train_data_filtered', type=int, default=0)
    parser.add_argument('--dev_data_name', type=str, default='hotpot_dev_distractor_wiki_tokenized.json')
    parser.add_argument('--pretrained_cfg_name', default=PRE_TAINED_LONFORMER_BASE, type=str)
    parser.add_argument('--gpu_num', default=4, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('--gamma', default=2.0, type=float, help='parameter for focal loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='parameter for focal loss')
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--score_model_name', default='MLP', type=str) #'DotProduct', 'BiLinear', 'MLP'
    parser.add_argument('--hop_model_name', default='DotProduct', type=str)  # 'DotProduct', 'BiLinear'
    parser.add_argument('--frozen_layer_num', default=0, type=int, help='number of layers for document encoder frozen during training')
    parser.add_argument('--project_dim', default=256, type=int)
    parser.add_argument('--global_mask_type', default='query_doc_sent', type=str) ## query, query_doc, query_doc_sent
    parser.add_argument('--training_shuffle', default=0, type=int)  ## whether re-order training data
    parser.add_argument('--sent_threshold', default=0.9, type=float)
    parser.add_argument('--doc_threshold', default=0.9, type=float)
    parser.add_argument('--max_sent_num', default=150, type=int)
    parser.add_argument('--max_doc_num', default=10, type=int)
    parser.add_argument('--accumulation_steps', default=0, type=int)
    parser.add_argument('--max_ctx_len', default=4096, type=int)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.00004, type=float) # 1e-5 level
    parser.add_argument('--batch_size', default=2, type=int)
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##+++++++++++++++++++++++++Parameter for multi-head self-attention++++++++++++++++++++++++++++++++++
    parser.add_argument('--input_drop', default=0.1, type=float)
    parser.add_argument('--attn_drop', default=0.1, type=float)
    parser.add_argument('--heads', default=8, type=float)
    parser.add_argument('--with_graph', default=1, type=int)
    parser.add_argument('--task', default='doc', type=str) ## doc, doc_sent, doc_sent_ans
    parser.add_argument('--with_graph_training', default=1, type=int)
    parser.add_argument('--span_weight', default=0.2, type=float)
    parser.add_argument('--pair_score_weight', default=1.0, type=float)
    parser.add_argument('--seq_project', default=True, action='store_true', help='whether perform sequence projection')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--num_labels', default=2, type=int, help='span prediction label') ##start and end position prediction, seperately
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
    date_time_str = get_date_time()
    if args.do_train:
        log_file = os.path.join(args.save_path, date_time_str + '_train.log')
    else:
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
    set_seeds(args.rand_seed)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be chosed.')

    if args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be chosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained reasonModel?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ########+++++++++++++++++++++++++++++
    abs_path = os.path.abspath(args.data_path)
    args.data_path = abs_path
    ########+++++++++++++++++++++++++++++
    # Write logs to checkpoint and console
    set_logger(args)
    if args.cuda:
        if args.do_debug:
            if args.gpu_num > 1:
                device_ids, used_memory = gpu_setting(args.gpu_num)
            else:
                device_ids, used_memory = gpu_setting()
            if used_memory > 100:
                logging.info('Using memory = {}'.format(used_memory))
            if device_ids is not None:
                device = torch.device('cuda:{}'.format(device_ids[0]))
            else:
                device = torch.device('cuda:0')
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
                    device_ids = []
                    for i in range(args.gpu_num - 1, -1, -1):
                        device_ids.append(i)
                        device = torch.device("cuda:%d" % i)
                else:
                    device_ids = list(range(args.gpu_num))
                    device = torch.device("cuda:0")
            else:
                device = torch.device("cuda:0")
                device_ids = None
                logging.info('Single GPU setting')
    else:
        device = torch.device('cpu')
        device_ids = None
        logging.info('CPU setting')

    logging.info('Loading training data...')
    train_data_loader, train_data_size = get_train_data_loader(args=args)
    estimated_max_steps = args.epoch * ((train_data_size // args.batch_size) + 1)
    if estimated_max_steps > args.max_steps:
        args.max_steps = args.epoch * ((train_data_size // args.batch_size) + 1)
    else:
        estimated_epoch = args.max_steps // ((train_data_size // args.batch_size) + 1)
        args.epoch = estimated_epoch
    logging.info('Loading development data...')
    dev_data_loader, _ = get_dev_data_loader(args=args)
    logging.info('Loading data completed')
    logging.info('*'*75)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if args.do_train:
        # Set training configuration
        start_time = time()
        logging.info('Loading reasonModel...')
        if args.init_checkpoint is None:
            model = get_model(args=args).to(device)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        else:
            model, optimizer = get_check_point(args=args)
            model = model.to(device)
        if device_ids is not None:
            model = DataParallel(model, device_ids=device_ids)
        logging.info('Model Parameter Configuration:')
        for name, param in model.named_parameters():
            logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
        logging.info('*' * 75)
        logging.info("Model hype-parameter information...")
        for key, value in vars(args).items():
            logging.info('Hype-parameter\t{} = {}'.format(key, value))
        logging.info('*' * 75)
        logging.info('batch_size = {}'.format(args.batch_size))
        logging.info('projection_dim = {}'.format(args.project_dim))
        logging.info('learning_rate = {}'.format(args.learning_rate))
        logging.info('Start training...')
        train_all_steps(model=model, optimizer=optimizer, dev_dataloader=dev_data_loader,
                        train_dataloader=train_data_loader, args=args)
        logging.info('Completed training in {:.4f} seconds'.format(time() - start_time))
        logging.info('Evaluating on Valid Dataset...')
        metric_dict = test_all_steps(model=model, test_data_loader=dev_data_loader, args=args)
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
                                       date_time_str + '_final_acc_' + answer_type_acc + '.json')
        dev_data_frame.to_json(dev_result_name, orient='records')
        logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
        logging.info('*' * 75)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++
        model_save_path = save_check_point(model=model, optimizer=optimizer, step='all_step', loss='final_loss',
                                     eval_metric='final', args=args)
        logging.info('Saving the mode in {}'.format(model_save_path))
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    main(parse_args())