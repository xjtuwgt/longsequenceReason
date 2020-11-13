import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import logging
import os
import torch
from multihopUtils.gpu_utils import gpu_setting
from distributedMultihopQA.parQATrainFunction import run_train_and_dev
from multihopQA.qarun import parse_args, set_logger

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