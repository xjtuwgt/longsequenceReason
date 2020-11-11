import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from multihopUtils.hotpotqaIOUtils import save_check_point, load_check_point
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
import os
from time import time
import torch
from modelTrain.QATrainFunction import log_metrics, test_all_steps, get_date_time

def training_warm_up(model, optimizer, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warm_up_steps
    start_time = time()
    step = 0
    training_logs = []
    logging.info('Starting warm up...')
    logging.info('*' * 75)
    #########
    model.train()
    model.zero_grad()
    #########
    for batch_idx, train_sample in enumerate(train_dataloader):
        loss, log = single_loss_computation(model=model, train_sample=train_sample, args=args)
        loss = loss / args.accumulation_steps  # Normalize our loss (if averaged)
        log = dict([(key, loss/args.accumulation_steps) for key, loss in log.items()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
        ########
        step = step + 1
        training_logs.append(log)
        if step % args.accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Training average', step, metrics)
            logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, 'warm_up', batch_idx + 1,
                                                                                     time() - start_time))
            training_logs = []
        if step >= warm_up_steps:
            logging.info('Warm up completed in {:.4f} seconds'.format(time() - start_time))
            logging.info('*' * 75)
            break
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, args=args)
    logging.info('*' * 75)
    logging.info('Answer type prediction accuracy: {}'.format(metric_dict['answer_type_acc']))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metrics'):
            logging.info('{} prediction'.format(key))
            log_metrics('Valid', 'warm up', value)
        logging.info('*' * 75)

def train_all_steps(model, optimizer, train_dataloader, dev_dataloader, args):
    assert args.save_checkpoint_steps % args.valid_steps == 0
    warm_up_steps = args.warm_up_steps
    if warm_up_steps > 0:
        training_warm_up(model=model, optimizer=optimizer, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, args=args)
        logging.info('*' * 75)
        current_learning_rate = optimizer.param_groups[-1]['lr']
        # learning_rate = args.learning_rate * 0.5
        learning_rate = current_learning_rate * 0.5
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logging.info('Update learning rate from {} to {}'.format(current_learning_rate, learning_rate))
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_steps, eta_min=1e-12)
    start_time = time()
    train_loss = 0.0
    eval_metric = None
    max_sent_pred_f1 = 0.0
    step = 0
    training_logs = []
    #########
    model.train()
    model.zero_grad()
    #########
    for epoch in range(1, args.epoch + 1):
        for batch_idx, train_sample in enumerate(train_dataloader):
            #########################
            loss, log = single_loss_computation(model=model, train_sample=train_sample, args=args)
            loss = loss / args.accumulation_steps  # Normalize our loss (if averaged)
            log = dict([(key, loss / args.accumulation_steps) for key, loss in log.items()])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
            #########################
            step = step + 1
            training_logs.append(log)
            if step % args.accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                # ##+++++++++++++++++++++++++++++++++++++++++++++++
                scheduler.step()
                # ##+++++++++++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.save_checkpoint_steps == 0:
                save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss, eval_metric=eval_metric, args=args)
                logging.info('Saving the mode in {}'.format(save_path))
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch, batch_idx + 1, time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = test_all_steps(model=model, test_data_loader=dev_dataloader, args=args)
                logging.info('*' * 75)
                answer_type_acc = metric_dict['answer_type_acc']
                eval_metric = answer_type_acc
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                sent_pred_f1 = metric_dict['supp_sent_metrics']['sp_f1']
                logging.info('*' * 75)
                # log_metrics('Valid', step, metric_dict['metrics'])
                for key, value in metric_dict.items():
                    if key.endswith('metrics'):
                        logging.info('{} prediction'.format(key))
                        log_metrics('Valid', step, value)
                logging.info('Answer type prediction accuracy: {}'.format(answer_type_acc))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                dev_data_frame = metric_dict['res_dataframe']
                date_time_str = get_date_time()
                dev_result_name = os.path.join(args.save_path,
                                               date_time_str + '_' + str(step) + "_acc_" + answer_type_acc + '.json')
                dev_data_frame.to_json(dev_result_name, orient='records')
                logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
                logging.info('*' * 75)
                ##++++++++++++++++++++++++++++++++++++++++++++++++++++
                if max_sent_pred_f1 < sent_pred_f1:
                    max_sent_pred_f1 = sent_pred_f1
                    save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss,
                                                 eval_metric=max_sent_pred_f1, args=args)
                    logging.info('Saving the mode in {} with current best metric = {:.4f}'.format(save_path, max_sent_pred_f1))
                #########
                model.train()
                model.zero_grad()
                #########
    logging.info('Training completed...')

def single_loss_computation(model, train_sample, args):
    if args.cuda:
        sample = dict()
        for key, value in train_sample.items():
            sample[key] = value.cuda()
    else:
        sample = train_sample
    loss_output = model(sample)
    yn_loss, span_loss, supp_doc_loss, supp_sent_loss = loss_output['yn_loss'], \
                                                        loss_output['span_loss'], \
                                                        loss_output['doc_loss'], loss_output['sent_loss']
    supp_doc_pair_loss = loss_output['doc_pair_loss']
    if args.do_retrieval:
        loss = supp_doc_loss + supp_sent_loss + supp_doc_pair_loss * args.pair_score_weight
    else:
        loss = supp_doc_loss + supp_sent_loss + span_loss * args.span_weight + yn_loss + supp_doc_pair_loss * args.pair_score_weight

    loss = loss.mean()
    log = {
        'al_loss': loss.mean().item(),
        'an_loss': span_loss.mean().item(),
        'sd_loss': supp_doc_loss.mean().item(),
        'pd_loss': supp_doc_pair_loss.mean().item(),
        'ss_loss': supp_sent_loss.mean().item(),
        'yn_loss': yn_loss.mean().item()
    }
    return loss, log
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++