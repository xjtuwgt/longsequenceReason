# # import os
# # import sys
# # PACKAGE_PARENT = '..'
# # SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# # sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# # from multihopUtils.hotpotqaIOUtils import save_check_point, load_check_point
# # from torch.optim.lr_scheduler import CosineAnnealingLR
# # import logging
# # import os
# # import pandas as pd
# # from time import time
# # import torch
# # from torch import Tensor as T
# # from torch.utils.data import DataLoader
# # from multihopQA.hotpotQAdataloader import HotpotTrainDataset, HotpotDevDataset
# # from multihopUtils.longformerQAUtils import LongformerQATensorizer, LongformerEncoder, get_hotpotqa_longformer_tokenizer
# # from reasonModel.UnifiedQAModel import LongformerHotPotQAModel
# # from pandas import DataFrame
# # from datetime import date, datetime
# # ##
# # MASK_VALUE = -1e9
# # ##
# #
# #
# # def test_all_steps_hierartical(model, test_data_loader, args):
# #     '''
# #             Evaluate the reasonModel on test or valid datasets
# #     '''
# #     model.eval()
# #     ###########################################################
# #     start_time = time()
# #     test_dataset = test_data_loader
# #     doc_logs, topk_sent_logs, thresh_sent_logs = [], [], []
# #     step = 0
# #     N = 0
# #     total_steps = len(test_dataset)
# #     # **********************************************************
# #     support_doc_pred_results, support_doc_score_results = [], []
# #     topk_support_sent_pred_results, topk_support_sent_score_results, topk_support_sent_doc_sent_pair_results = [], [], []
# #     thresh_support_sent_pred_results, thresh_support_sent_score_results, thresh_support_sent_doc_sent_pair_results = [], [], []
# #     answer_type_pred_results = []
# #     topk_span_results = []
# #     threhold_span_results = []
# #     encode_id_results = []
# #     correct_answer_num = 0
# #     # **********************************************************
# #     with torch.no_grad():
# #         for test_sample in test_dataset:
# #             if args.cuda:
# #                 sample = dict()
# #                 for key, value in test_sample.items():
# #                     sample[key] = value.cuda()
# #             else:
# #                 sample = test_sample
# #             output = model(sample)
# #             N = N + sample['doc_labels'].shape[0]
# #             eval_res = hierartical_metric_computation(output_scores=output, sample=sample, args=args)
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             # ******************************************
# #             step += 1
# #             if step % args.test_log_steps == 0:
# #                 logging.info('Evaluating the reasonModel... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
# #     doc_metrics, topk_sent_metrics, thresh_sent_metrics = {}, {}, {}
# #     for metric in doc_logs[0].keys():
# #         doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
# #     for metric in topk_sent_logs[0].keys():
# #         topk_sent_metrics[metric] = sum([log[metric] for log in topk_sent_logs]) / len(topk_sent_logs)
# #     for metric in thresh_sent_logs[0].keys():
# #         thresh_sent_metrics[metric] = sum([log[metric] for log in thresh_sent_logs]) / len(thresh_sent_logs)
# #     ##=================================================
# #     # answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
# #     # result_dict = {'aty_pred': answer_type_pred_results,
# #     #                'sd_pred': support_doc_pred_results,
# #     #                'ss_pred': support_sent_pred_results,
# #     #                'sps_pred': span_pred_start_results,
# #     #                'spe_pred': span_pred_end_results,
# #     #                'sd_score': support_doc_score_results,
# #     #                'ss_score': support_sent_score_results,
# #     #                'ss_ds_pair': support_sent_doc_sent_pair_results,
# #     #                'encode_ids': encode_id_results} ## for detailed results checking
# #     # res_data_frame = DataFrame(result_dict)
# #     # ##=================================================
# #     # return {'supp_doc_metrics': doc_metrics, 'supp_sent_metrics': sent_metrics,
# #     #         'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}
# #
# #
# # def hierartical_metric_computation(output_scores: dict, sample: dict, args):
# #     #'yn_score', 'span_score', 'doc_score': (supp_doc_scores, supp_head_tail_scores), 'sent_score':
# #     yn_scores = output_scores['yn_score']
# #     yn_true_labels = sample['yes_no']
# #     if len(yn_true_labels.shape) > 1:
# #         yn_true_labels = yn_true_labels.squeeze(dim=-1)
# #     yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
# #     correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
# #     yn_predicted_labels = yn_predicted_labels.detach().tolist()
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     doc_scores, _ = output_scores['doc_score']
# #     doc_mask = sample['doc_lens']
# #     true_doc_labels = sample['doc_labels']
# #
# #     #############################################################
# #     doc_start_position, doc_end_position = sample['doc_start'], sample['doc_end'] ## doc start and end position for answer span prediction
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     sent2doc_map = sample['s2d_map'] ## restore document ids
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     doc_res_dict = supp_doc_prediction(doc_scores=doc_scores, mask=doc_mask,
# #                                        labels=true_doc_labels, doc_start_pos=doc_start_position,
# #                                        doc_end_pos=doc_end_position, sent2doc_map=sent2doc_map, top_k=3, threshold=args.doc_threshold)
# #     #################################################################
# #     sent_scores = output_scores['sent_scores']
# #     sentIndoc_map = sample['sInd_map']
# #     topk_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['top_k_sents'])
# #     threshold_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['threshold_sents'])
# #     true_sent_labels = sample['sent_labels']
# #     sent_lens = sample['sent_lens']
# #
# #     topk_sent_res_dict = supp_sent_predictions(scores=topk_sent_scores, labels=true_sent_labels,
# #                                                mask=sent_lens, sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map,
# #                                                pred_num=2, threshold=args.sent_threshold)
# #     threshold_sent_res_dict = supp_sent_predictions(scores=threshold_sent_scores, labels=true_sent_labels, mask=sent_lens,
# #                                                     sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map, pred_num=2,
# #                                                     threshold=args.sent_threshold)
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     topk_span_start_predictions, topk_span_end_predictions = [], []
# #     threshold_span_start_predictions, threshold_span_end_predictions = [], []
# #     span_start_scores, span_end_scores = output_scores['span_score']
# #     topk_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
# #     topk_span_start_i = torch.argmax(topk_span_start_scores, dim=-1)
# #     topk_span_start_predictions.append(topk_span_start_i)
# #
# #     topk_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
# #     topk_span_end_i = torch.argmax(topk_span_end_scores, dim=-1)
# #     topk_span_end_predictions.append(topk_span_end_i)
# #
# #     threshold_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
# #     threshold_span_start_i = torch.argmax(threshold_span_start_scores)
# #     threshold_span_start_predictions.append(threshold_span_start_i)
# #     threshold_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
# #     threshold_span_end_i = torch.argmax(threshold_span_end_scores)
# #     threshold_span_end_predictions.append(threshold_span_end_i)
# #
# #     topk_span_start_end_pair = list(zip(topk_span_start_predictions, topk_span_end_predictions))
# #     threshold_span_start_end_pair = list(zip(threshold_span_start_predictions, threshold_span_end_predictions))
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     encode_ids = sample['ctx_encode'].detach().tolist()
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     res = {'answer_type_pred': (correct_yn, yn_predicted_labels),
# #            'supp_doc_pred': doc_res_dict,
# #            'topk_sent_pred': topk_sent_res_dict,
# #            'threshold_sent_pred': threshold_sent_res_dict,
# #            'topk_span_pred': topk_span_start_end_pair,
# #            'threhold_span_pred': threshold_span_start_end_pair,
# #            'encode_id': encode_ids}
# #     return res
# #
# # def sent_score_extraction(sent_scores: T, doc2sent_idexes: list):
# #     batch_size, sent_num = sent_scores.shape
# #     temp_scores = torch.empty(batch_size, sent_num).fill_(MASK_VALUE).to(sent_scores.device)
# #     for idx in range(batch_size):
# #         temp_scores[idx, doc2sent_idexes[idx]] = sent_scores[idx, doc2sent_idexes[idx]]
# #     return temp_scores
# #
# # def token_score_extraction(token_scores: T, doc_start_end_pair_list: list):
# #     batch_size, seq_num = token_scores.shape
# #     temp_scores = torch.empty(batch_size, seq_num).fill_(MASK_VALUE).to(token_scores.device)
# #     for idx in range(batch_size):
# #         doc_start_end_i = doc_start_end_pair_list[idx]
# #         for dox_idx, doc_pair in enumerate(doc_start_end_i):
# #             start_i, end_i = doc_pair
# #             temp_scores[idx][start_i:(end_i+1)] = token_scores[idx][start_i:(end_i+1)]
# #     return temp_scores
# #
# # def sp_score(prediction, gold):
# #     cur_sp_pred = set(prediction)
# #     gold_sp_pred = set(gold)
# #     tp, fp, fn = 0, 0, 0
# #     for e in cur_sp_pred:
# #         if e in gold_sp_pred:
# #             tp += 1
# #         else:
# #             fp += 1
# #     for e in gold_sp_pred:
# #         if e not in cur_sp_pred:
# #             fn += 1
# #     prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
# #     recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
# #     f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
# #     em = 1.0 if fp + fn == 0 else 0.0
# #     return em, prec, recall, f1
# #
# # def supp_sent_predictions(scores: T, labels: T, mask: T, sent2doc_map: T, sentIndoc_map: T, pred_num=2, threshold=0.8):
# #     batch_size, sample_size = scores.shape[0], scores.shape[1]
# #     scores = torch.sigmoid(scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1)
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     logs = []
# #     predicted_labels = []
# #     true_labels = []
# #     score_list = []
# #     doc_sent_pair_list = []
# #     for idx in range(batch_size):
# #         score_list.append(masked_scores[idx].detach().tolist())
# #         # ==================
# #         doc_fact_i = sent2doc_map[idx].detach().tolist()
# #         sent_fact_i = sentIndoc_map[idx].detach().tolist()
# #         doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
# #         doc_sent_pair_list.append(doc_sent_pair_i)
# #         # ==================
# #         pred_idxes_i = argsort[idx].tolist()
# #         pred_labels_i = pred_idxes_i[:pred_num]
# #         for i in range(pred_num, sample_size):
# #             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
# #                 pred_labels_i.append(pred_idxes_i[i])
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
# #         # +++++++++++++++++
# #         predicted_labels.append(pred_labels_i)
# #         true_labels.append(labels_i)
# #         # +++++++++++++++++
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
# #         logs.append({
# #             'sp_em': em_i,
# #             'sp_f1': f1_i,
# #             'sp_prec': prec_i,
# #             'sp_recall':recall_i
# #         })
# #     res = {'log': logs, 'prediction': predicted_labels, 'score': score_list, 'doc_sent_pair': doc_sent_pair_list}
# #     return res
# #
# #
# # def supp_doc_prediction(doc_scores: T, labels: T, mask: T, doc_start_pos: T, doc_end_pos: T, sent2doc_map, top_k=2, threshold=0.9):
# #     batch_size, doc_num = doc_scores.shape
# #     top_k_predictions = []
# #     threshold_predictions = []
# #     ####
# #     top_k_doc_start_end = []
# #     top_k_sent_idxes = []
# #     #############################################
# #     threshold_doc_start_end = []
# #     threshold_sent_idxes = []
# #     ############################################
# #     scores = torch.sigmoid(doc_scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1)
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     logs = []
# #     true_labels = []
# #     score_list = []
# #     for idx in range(batch_size):
# #         score_list.append(masked_scores[idx].detach().tolist())
# #         # ==================
# #         pred_idxes_i = argsort[idx].tolist()
# #         top_k_labels_i = pred_idxes_i[:top_k]
# #         threhold_labels_i = pred_idxes_i[:top_k]
# #         # ==================
# #         for i in range(top_k, doc_num):
# #             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[top_k - 1]]:
# #                 threhold_labels_i.append(pred_idxes_i[i])
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist()
# #         # +++++++++++++++++
# #         top_k_predictions.append(top_k_labels_i)
# #         threshold_predictions.append(threhold_labels_i)
# #         true_labels.append(labels_i)
# #         # +++++++++++++++++
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=top_k_labels_i, gold=labels_i)
# #         t_em_i, t_prec_i, t_recall_i, t_f1_i = sp_score(prediction=threhold_labels_i, gold=labels_i)
# #         logs.append({
# #             'sp_em': em_i,
# #             'sp_f1': f1_i,
# #             'sp_prec': prec_i,
# #             'sp_recall': recall_i,
# #             'threshold_sp_em': t_em_i,
# #             'threshold_sp_f1': t_f1_i,
# #             'threshold_sp_prec': t_prec_i,
# #             'threshold_sp_recall': t_recall_i,
# #         })
# #         ###############
# #         top_k_start_end_i = []
# #         top_k_sent_i = []
# #         threshold_start_end_i = []
# #         threshold_sent_i = []
# #         for topk_pre_doc_idx in top_k_labels_i:
# #             doc_s_i = doc_start_pos[idx][topk_pre_doc_idx].data.item()
# #             doc_e_i = doc_end_pos[idx][topk_pre_doc_idx].data.item()
# #             top_k_start_end_i.append((doc_s_i, doc_e_i))
# #             sent_idx_i = (sent2doc_map[idx] == topk_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
# #             top_k_sent_i += sent_idx_i
# #
# #         for thresh_pre_doc_idx in threhold_labels_i:
# #             doc_s_i = doc_start_pos[idx][thresh_pre_doc_idx].data.item()
# #             doc_e_i = doc_end_pos[idx][thresh_pre_doc_idx].data.item()
# #             threshold_start_end_i.append((doc_s_i, doc_e_i))
# #             sent_idx_i = (sent2doc_map[idx] == thresh_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
# #             threshold_sent_i += sent_idx_i
# #         ###############
# #         top_k_doc_start_end.append(top_k_start_end_i)
# #         threshold_doc_start_end.append(threshold_start_end_i)
# #         top_k_sent_idxes.append(top_k_sent_i)
# #         threshold_sent_idxes.append(threshold_sent_i)
# #
# #     res = {'log': logs,
# #            'top_k_doc': top_k_predictions,
# #            'threshold_doc': threshold_predictions,
# #            'true_doc': true_labels,
# #            'score': score_list,
# #            'top_k_doc2token': top_k_doc_start_end,
# #            'top_k_sents': top_k_sent_idxes,
# #            'threshold_doc2token': threshold_doc_start_end,
# #            'threshold_sents': threshold_sent_idxes}
# #     return res
# #
# # def test_all_steps(model, test_data_loader, device, args):
# #     '''
# #             Evaluate the reasonModel on test or valid datasets
# #     '''
# #     model.eval()
# #     ###########################################################
# #     start_time = time()
# #     test_dataset = test_data_loader
# #     doc_logs, sent_logs = [], []
# #     step = 0
# #     N = 0
# #     total_steps = len(test_dataset)
# #     # **********************************************************
# #     support_doc_pred_results = []
# #     support_sent_pred_results, support_sent_doc_sent_pair_results = [], []
# #     answer_type_pred_results = []
# #     span_pred_results = []
# #     encode_id_results = []
# #     correct_answer_num = 0
# #     # **********************************************************
# #     with torch.no_grad():
# #         for test_sample in test_dataset:
# #             if args.cuda:
# #                 sample = dict()
# #                 for key, value in test_sample.items():
# #                     sample[key] = value.to(device)
# #             else:
# #                 sample = test_sample
# #             output = model(sample)
# #             N = N + sample['doc_labels'].shape[0]
# #             eval_res = metric_computation(output_scores=output, sample=sample, args=args)
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             correct_yn, yn_predicted_labels = eval_res['answer_type']
# #             correct_answer_num += correct_yn
# #             answer_type_pred_results += yn_predicted_labels
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             span_predicted_i = eval_res['answer_span']
# #             span_pred_results += span_predicted_i
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             encode_ids = eval_res['encode_ids']
# #             encode_id_results += encode_ids
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             doc_metric_logs, doc_pred_res = eval_res['supp_doc']
# #             doc_logs += doc_metric_logs
# #             doc_predicted_labels = doc_pred_res
# #             support_doc_pred_results += doc_predicted_labels
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             sent_metric_logs, sent_pred_res = eval_res['supp_sent']
# #             sent_logs += sent_metric_logs
# #             sent_predicted_labels, doc_sent_fact_pair = sent_pred_res
# #             support_sent_pred_results += sent_predicted_labels
# #             support_sent_doc_sent_pair_results += doc_sent_fact_pair
# #             # +++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             # ******************************************
# #             step += 1
# #             if step % args.test_log_steps == 0:
# #                 logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
# #     doc_metrics, sent_metrics = {}, {}
# #     for metric in doc_logs[0].keys():
# #         doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
# #     for metric in sent_logs[0].keys():
# #         sent_metrics[metric] = sum([log[metric] for log in sent_logs]) / len(sent_logs)
# #     ##=================================================
# #     answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
# #     result_dict = {'aty_pred': answer_type_pred_results,
# #                    'sd_pred': support_doc_pred_results,
# #                    'ss_pred': support_sent_pred_results,
# #                    'ans_span': span_pred_results,
# #                    'ss_ds_pair': support_sent_doc_sent_pair_results,
# #                    'encode_ids': encode_id_results} ## for detailed results checking
# #     res_data_frame = DataFrame(result_dict)
# #     ##=================================================
# #     return {'supp_doc_metrics': doc_metrics, 'supp_sent_metrics': sent_metrics,
# #             'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}
# #
# # def metric_computation(output_scores: dict, sample: dict, args):
# #     # =========Answer type prediction==========================
# #     yn_scores = output_scores['yn_score']
# #     yn_true_labels = sample['yes_no']
# #     if len(yn_true_labels.shape) > 1:
# #         yn_true_labels = yn_true_labels.squeeze(dim=-1)
# #     yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
# #     correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
# #     yn_predicted_labels = yn_predicted_labels.detach().tolist()
# #     # =========Answer span prediction==========================
# #     start_logits, end_logits = output_scores['span_score']
# #     predicted_span_start = torch.argmax(start_logits, dim=-1)
# #     predicted_span_end = torch.argmax(end_logits, dim=-1)
# #     predicted_span_start = predicted_span_start.detach().tolist()
# #     predicted_span_end = predicted_span_end.detach().tolist()
# #     predicted_span_pair = list(zip(predicted_span_start, predicted_span_end))
# #     ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     # +++++++++ supp doc prediction +++++++++++++++++++++++++++
# #     doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
# #     doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
# #     supp_doc_scores, _ = output_scores['doc_score']
# #     doc_metric_logs, doc_pred_res = support_doc_infor_evaluation(scores=supp_doc_scores, labels=doc_label, mask=doc_mask, pred_num=2)
# #     # +++++++++ supp doc prediction +++++++++++++++++++++++++++
# #     # +++++++++ supp sent prediction +++++++++++++++++++++++++++
# #     supp_sent_scores = output_scores['sent_score']
# #     sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
# #     sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
# #     sent_fact_doc_idx, sent_fact_sent_idx = sample['s2d_map'], sample['sInd_map']
# #     sent_metric_logs, sent_pred_res = support_sent_infor_evaluation(scores=supp_sent_scores, labels=sent_label, mask=sent_mask, pred_num=2,
# #                                                                threshold=args.sent_threshold, doc_fact=sent_fact_doc_idx, sent_fact=sent_fact_sent_idx)
# #     # +++++++++ supp sent prediction +++++++++++++++++++++++++++
# #     # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
# #     encode_ids = sample['ctx_encode'].detach().tolist()
# #     # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
# #     return {'answer_type': (correct_yn, yn_predicted_labels),
# #             'answer_span': predicted_span_pair,
# #             'supp_doc': (doc_metric_logs, doc_pred_res),
# #             'supp_sent': (sent_metric_logs, sent_pred_res),
# #             'encode_ids': encode_ids}
# #
# # def support_doc_infor_evaluation(scores: T, labels: T, mask: T, pred_num=2):
# #     batch_size, sample_size = scores.shape[0], scores.shape[1]
# #     scores = torch.sigmoid(scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1)
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     logs = []
# #     predicted_labels = []
# #     score_list = []
# #     for idx in range(batch_size):
# #         score_list.append(masked_scores[idx].detach().tolist())
# #         pred_idxes_i = argsort[idx].tolist()
# #         pred_labels_i = pred_idxes_i[:pred_num]
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
# #         # +++++++++++++++++
# #         predicted_labels.append(pred_labels_i)
# #         # +++++++++++++++++
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
# #         logs.append({
# #             'sp_em': em_i,
# #             'sp_f1': f1_i,
# #             'sp_prec': prec_i,
# #             'sp_recall':recall_i
# #         })
# #     return logs, predicted_labels
# # ####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #
# # def support_sent_infor_evaluation(scores: T, labels: T, mask: T, doc_fact: T, sent_fact: T, pred_num=2, threshold=0.8):
# #     batch_size, sample_size = scores.shape[0], scores.shape[1]
# #     scores = torch.sigmoid(scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1)
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     logs = []
# #     predicted_labels = []
# #     doc_sent_pair_list = []
# #     for idx in range(batch_size):
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         doc_fact_i = doc_fact[idx].detach().tolist()
# #         sent_fact_i = sent_fact[idx].detach().tolist()
# #         doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
# #         doc_sent_pair_list.append(doc_sent_pair_i)
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         pred_idxes_i = argsort[idx].tolist()
# #         pred_labels_i = pred_idxes_i[:pred_num]
# #         for i in range(pred_num, sample_size):
# #             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
# #                 pred_labels_i.append(pred_idxes_i[i])
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         predicted_labels.append(pred_labels_i)
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
# #         logs.append({
# #             'sp_em': em_i,
# #             'sp_f1': f1_i,
# #             'sp_prec': prec_i,
# #             'sp_recall':recall_i
# #         })
# #     return logs, (predicted_labels, doc_sent_pair_list)
# # ####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #
# # def sp_score(prediction, gold):
# #     cur_sp_pred = set(prediction)
# #     gold_sp_pred = set(gold)
# #     tp, fp, fn = 0, 0, 0
# #     for e in cur_sp_pred:
# #         if e in gold_sp_pred:
# #             tp += 1
# #         else:
# #             fp += 1
# #     for e in gold_sp_pred:
# #         if e not in cur_sp_pred:
# #             fn += 1
# #     prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
# #     recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
# #     f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
# #     em = 1.0 if fp + fn == 0 else 0.0
# #     return em, prec, recall, f1
# #
# #     # print('ans {}\n topk sd {}\n th sd {}\n k ss {}\nt ss {}\n k pair {}\n t pair {} \n k '
# #     #       'ans {}\n t ans {}\n encode {}\n'.format(len(answer_type_pred_results),
# #     #                                                    len(topk_support_doc_pred_results),
# #     #                                                    len(threshold_support_doc_pred_results),
# #     #                                                len(topk_support_sent_pred_results),
# #     #                                                len(thresh_support_sent_pred_results),
# #     #                                                len(topk_support_sent_doc_sent_pair_results),
# #     #                                                len(thresh_support_sent_doc_sent_pair_results),
# #     #                                                len(topk_answer_span_results),
# #     #                                                    len(thresh_answer_span_results),
# #     #                                                    len(encode_id_results)))
# #
# #
# # # dev_data_frame = metric_dict['res_dataframe']
# # # date_time_str = get_date_time()
# # # dev_result_name = os.path.join(args.save_path,
# # #                                date_time_str + '_final_acc_' + answer_type_acc + '.json')
# # # dev_data_frame.to_json(dev_result_name, orient='records')
# # # logging.info('Saving {} record results to {}'.format(dev_data_frame.shape, dev_result_name))
# # # logging.info('*' * 75)
# #
# # # def metric_computation(output_scores: dict, sample: dict, args):
# # #     # =========Answer type prediction==========================
# # #     yn_scores = output_scores['yn_score']
# # #     yn_true_labels = sample['yes_no']
# # #     if len(yn_true_labels.shape) > 1:
# # #         yn_true_labels = yn_true_labels.squeeze(dim=-1)
# # #     yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
# # #     correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
# # #     yn_predicted_labels = yn_predicted_labels.detach().tolist()
# # #     # =========Answer span prediction==========================
# # #     start_logits, end_logits = output_scores['span_score']
# # #     predicted_span_start = torch.argmax(start_logits, dim=-1)
# # #     predicted_span_end = torch.argmax(end_logits, dim=-1)
# # #     predicted_span_start = predicted_span_start.detach().tolist()
# # #     predicted_span_end = predicted_span_end.detach().tolist()
# # #     predicted_span_pair = list(zip(predicted_span_start, predicted_span_end))
# # #     ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # #     # +++++++++ supp doc prediction +++++++++++++++++++++++++++
# # #     doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
# # #     doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
# # #     supp_doc_scores, _ = output_scores['doc_score']
# # #     doc_metric_logs, doc_pred_res = support_doc_evaluation(scores=supp_doc_scores, labels=doc_label, mask=doc_mask, pred_num=2)
# # #     # +++++++++ supp doc prediction +++++++++++++++++++++++++++
# # #     # +++++++++ supp sent prediction +++++++++++++++++++++++++++
# # #     supp_sent_scores = output_scores['sent_score']
# # #     sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
# # #     sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
# # #     sent_fact_doc_idx, sent_fact_sent_idx = sample['s2d_map'], sample['sInd_map']
# # #     sent_metric_logs, _, sent_pred_res = support_sent_evaluation(scores=supp_sent_scores, labels=sent_label, mask=sent_mask, pred_num=2,
# # #                                                                threshold=args.sent_threshold, doc_fact=sent_fact_doc_idx, sent_fact=sent_fact_sent_idx)
# # #     # +++++++++ supp sent prediction +++++++++++++++++++++++++++
# # #     # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
# # #     encode_ids = sample['ctx_encode'].detach().tolist()
# # #     # +++++++++ encode ids +++++++++++++++++++++++++++++++++++++
# # #     return {'answer_type': (correct_yn, yn_predicted_labels),
# # #             'answer_span': predicted_span_pair,
# # #             'supp_doc': (doc_metric_logs, doc_pred_res),
# # #             'supp_sent': (sent_metric_logs, sent_pred_res),
# # #             'encode_ids': encode_ids}
# #
# #
# #
# # import os
# # import sys
# # PACKAGE_PARENT = '..'
# # SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# # sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# # import logging
# # from time import time
# # import torch
# # from pandas import DataFrame
# # from torch import Tensor as T
# # from modelEvaluation.hotpotEvaluationUtils import sp_score
# # ########################################################################################################################
# # MASK_VALUE = -1e9
# # ########################################################################################################################
# #
# # ########################################################################################################################
# # def hierartical_decoder(model, device, test_data_loader, doc_topk, args):
# #     '''
# #             Evaluate the reasonModel on test or valid datasets
# #     '''
# #     model.eval()
# #     ###########################################################
# #     start_time = time()
# #     test_dataset = test_data_loader
# #     doc_logs, topk_sent_logs, thresh_sent_logs = [], [], []
# #     step = 0
# #     N = 0
# #     total_steps = len(test_dataset)
# #     # **********************************************************
# #     topk_support_doc_pred_results, threshold_support_doc_pred_results, answer_type_pred_results = [], [], []
# #     topk_support_sent_pred_results, topk_support_sent_doc_sent_pair_results = [], []
# #     thresh_support_sent_pred_results, thresh_support_sent_doc_sent_pair_results = [], []
# #     topk_answer_span_results, thresh_answer_span_results = [], []
# #     correct_answer_num = 0
# #     encode_id_results = []
# #     # **********************************************************
# #     with torch.no_grad():
# #         for test_sample in test_dataset:
# #             if args.cuda:
# #                 sample = dict()
# #                 for key, value in test_sample.items():
# #                     sample[key] = value.to(device)
# #             else:
# #                 sample = test_sample
# #             output = model(sample)
# #             N = N + sample['doc_labels'].shape[0]
# #             eval_res = hierartical_metric_computation(output_scores=output, sample=sample, doc_topk=doc_topk, args=args)
# #             # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             correct_yn, yn_predicted_labels = eval_res['answer_type_pred']
# #             correct_answer_num = correct_answer_num + correct_yn
# #             answer_type_pred_results = answer_type_pred_results  + yn_predicted_labels
# #             # **********************************************************************************************************
# #             # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             doc_res_dict_i = eval_res['supp_doc_pred']
# #             doc_logs += doc_res_dict_i['log']
# #             topk_support_doc_pred_results += doc_res_dict_i['top_k_doc']
# #             threshold_support_doc_pred_results += doc_res_dict_i['threshold_doc']
# #             # **********************************************************************************************************
# #             # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             topk_sent_res_dict_i = eval_res['topk_sent_pred']
# #             topk_sent_logs += topk_sent_res_dict_i['log']
# #             topk_support_sent_pred_results += topk_sent_res_dict_i['prediction']
# #             topk_support_sent_doc_sent_pair_results += topk_sent_res_dict_i['doc_sent_pair']
# #             # **********************************************************************************************************
# #             # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             thresh_sent_res_dict_i = eval_res['threshold_sent_pred']
# #             thresh_sent_logs += thresh_sent_res_dict_i['log']
# #             thresh_support_sent_pred_results += thresh_sent_res_dict_i['prediction']
# #             thresh_support_sent_doc_sent_pair_results += thresh_sent_res_dict_i['doc_sent_pair']
# #             # **********************************************************************************************************
# #             # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #             topk_answer_span_results += eval_res['topk_span_pred']
# #             thresh_answer_span_results += eval_res['threshold_span_pred']
# #             encode_i = sample['ctx_encode'].detach().tolist()
# #             encode_id_results += encode_i
# #             # **********************************************************************************************************
# #             step += 1
# #             if step % args.test_log_steps == 0:
# #                 logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
# #
# #     ###############################################################
# #     doc_metrics, topk_sent_metrics, thresh_sent_metrics = {}, {}, {}
# #     for metric in doc_logs[0].keys():
# #         doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
# #     for metric in topk_sent_logs[0].keys():
# #         topk_sent_metrics[metric] = sum([log[metric] for log in topk_sent_logs]) / len(topk_sent_logs)
# #     for metric in thresh_sent_logs[0].keys():
# #         thresh_sent_metrics[metric] = sum([log[metric] for log in thresh_sent_logs]) / len(thresh_sent_logs)
# #     ##=================================================
# #     answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
# #     result_dict = {'aty_pred': answer_type_pred_results,
# #                    'topk_sd_pred': topk_support_doc_pred_results,
# #                    'thresh_sd_pred': threshold_support_doc_pred_results,
# #                    'topk_ss_pred': topk_support_sent_pred_results,
# #                    'thresh_ss_pred': threshold_support_doc_pred_results,
# #                    'topk_ds_pair': topk_support_sent_doc_sent_pair_results,
# #                    'thresh_ds_pair': thresh_support_sent_doc_sent_pair_results,
# #                    'topk_ans_span': topk_answer_span_results,
# #                    'thresh_ans_span': thresh_answer_span_results,
# #                    'encode_ids': encode_id_results}  ## for detailed results checking
# #     res_data_frame = DataFrame(result_dict)
# #     # ##=================================================
# #     return {'supp_doc_metrics': doc_metrics, 'topk_supp_sent_metrics': topk_sent_metrics,
# #             'thresh_supp_sent_metrics': thresh_sent_metrics,
# #             'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}
# #
# #
# # def hierartical_metric_computation(output_scores: dict, sample: dict, doc_topk, args):
# #     #'yn_score', 'span_score', 'doc_score': (supp_doc_scores, None), 'sent_score':
# #     yn_scores = output_scores['yn_score']
# #     yn_true_labels = sample['yes_no']
# #     if len(yn_true_labels.shape) > 1:
# #         yn_true_labels = yn_true_labels.squeeze(dim=-1)
# #     yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
# #     correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
# #     yn_predicted_labels = yn_predicted_labels.detach().tolist()
# #     ####################################################################################################################
# #     doc_scores, _ = output_scores['doc_score']
# #     doc_mask = sample['doc_lens']
# #     true_doc_labels = sample['doc_labels']
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     doc_start_position, doc_end_position = sample['doc_start'], sample['doc_end'] ## doc start and end position for answer span prediction
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     sent2doc_map = sample['s2d_map'] ## absolute sentence index map to document index, e.g., 150 sentence indexes to 10 documents
# #     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     doc_res_dict = hierartical_supp_doc_prediction(doc_scores=doc_scores, mask=doc_mask,
# #                                        labels=true_doc_labels, doc_start_pos=doc_start_position,
# #                                        doc_end_pos=doc_end_position, sent2doc_map=sent2doc_map, top_k=doc_topk, threshold=args.doc_threshold)
# #     ####################################################################################################################
# #     ####################################################################################################################
# #     ####################################################################################################################
# #     sent_scores = output_scores['sent_score']
# #     sentIndoc_map = sample['sInd_map'] ## absolute sentence index map to relative sentence index, e.g., 150 sentence indexes to 10 documents
# #     topk_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['top_k_sents'])
# #     threshold_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['threshold_sents'])
# #     true_sent_labels = sample['sent_labels']
# #     sent_lens = sample['sent_lens']
# #     topk_sent_res_dict = supp_sent_predictions(scores=topk_sent_scores, labels=true_sent_labels,
# #                                                mask=sent_lens, sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map,
# #                                                pred_num=2, threshold=args.sent_threshold)
# #     threshold_sent_res_dict = supp_sent_predictions(scores=threshold_sent_scores, labels=true_sent_labels, mask=sent_lens,
# #                                                     sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map, pred_num=2,
# #                                                     threshold=args.sent_threshold)
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     span_start_scores, span_end_scores = output_scores['span_score']
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     topk_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
# #     topk_span_start_i = torch.argmax(topk_span_start_scores, dim=-1)
# #     topk_span_start_i = topk_span_start_i.detach().tolist()
# #
# #     topk_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
# #     topk_span_end_i = torch.argmax(topk_span_end_scores, dim=-1)
# #     topk_span_end_i = topk_span_end_i.detach().tolist()
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     threshold_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
# #     threshold_span_start_i = torch.argmax(threshold_span_start_scores, dim=-1)
# #     threshold_span_start_i = threshold_span_start_i.detach().tolist()
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     threshold_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
# #     threshold_span_end_i = torch.argmax(threshold_span_end_scores, dim=-1)
# #     threshold_span_end_i = threshold_span_end_i.detach().tolist()
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     topk_span_start_end_pair = list(zip(topk_span_start_i, topk_span_end_i))
# #     threshold_span_start_end_pair = list(zip(threshold_span_start_i, threshold_span_end_i))
# #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #     res = {'answer_type_pred': (correct_yn, yn_predicted_labels),
# #            'supp_doc_pred': doc_res_dict,
# #            'topk_sent_pred': topk_sent_res_dict,
# #            'threshold_sent_pred': threshold_sent_res_dict,
# #            'topk_span_pred': topk_span_start_end_pair,
# #            'threshold_span_pred': threshold_span_start_end_pair}
# #     return res
# #
# # def sent_score_extraction(sent_scores: T, doc2sent_idexes: list):
# #     batch_size, sent_num = sent_scores.shape
# #     temp_scores = torch.empty(batch_size, sent_num).fill_(MASK_VALUE).to(sent_scores.device)
# #     for idx in range(batch_size):
# #         temp_scores[idx, doc2sent_idexes[idx]] = sent_scores[idx, doc2sent_idexes[idx]]
# #     return temp_scores
# #
# # def token_score_extraction(token_scores: T, doc_start_end_pair_list: list):
# #     batch_size, seq_num = token_scores.shape
# #     temp_scores = torch.empty(batch_size, seq_num).fill_(MASK_VALUE).to(token_scores.device)
# #     for idx in range(batch_size):
# #         doc_start_end_i = doc_start_end_pair_list[idx]
# #         for dox_idx, doc_pair in enumerate(doc_start_end_i):
# #             start_i, end_i = doc_pair
# #             temp_scores[idx][start_i:(end_i+1)] = token_scores[idx][start_i:(end_i+1)]
# #     return temp_scores
# #
# # def supp_sent_predictions(scores: T, labels: T, mask: T, sent2doc_map: T, sentIndoc_map: T, pred_num=2, threshold=0.9):
# #     batch_size, sample_size = scores.shape[0], scores.shape[1]
# #     scores = torch.sigmoid(scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1)
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     logs = []
# #     predicted_labels = []
# #     # true_labels = []
# #     doc_sent_pair_list = []
# #     for idx in range(batch_size):
# #         # ============================================================================================================
# #         doc_fact_i = sent2doc_map[idx].detach().tolist()
# #         sent_fact_i = sentIndoc_map[idx].detach().tolist()
# #         doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
# #         doc_sent_pair_list.append(doc_sent_pair_i) ## for final leadboard evaluation
# #         # ============================================================================================================
# #         pred_idxes_i = argsort[idx].tolist()
# #         pred_labels_i = pred_idxes_i[:pred_num]
# #         for i in range(pred_num, sample_size):
# #             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
# #                 pred_labels_i.append(pred_idxes_i[i])
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         predicted_labels.append(pred_labels_i)
# #         # true_labels.append(labels_i)
# #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
# #         logs.append({
# #             'sp_em': em_i,
# #             'sp_f1': f1_i,
# #             'sp_prec': prec_i,
# #             'sp_recall':recall_i
# #         })
# #     res = {'log': logs, 'prediction': predicted_labels, 'doc_sent_pair': doc_sent_pair_list}
# #     return res
# #
# #
# # def hierartical_supp_doc_prediction(doc_scores: T, labels: T, mask: T, doc_start_pos: T, doc_end_pos: T, sent2doc_map, top_k=2, threshold=0.9):
# #     batch_size, doc_num = doc_scores.shape
# #     top_k_predictions = []
# #     threshold_predictions = []
# #     ####################################################################################################################
# #     top_k_doc_start_end = [] ### for answer span prediction
# #     top_k_sent_idxes = [] ## for support sentence prediction
# #     ####################################################################################################################
# #     threshold_doc_start_end = [] ### for answer span prediction
# #     threshold_sent_idxes = [] ### for support sentence prediction
# #     ####################################################################################################################
# #     scores = torch.sigmoid(doc_scores)
# #     masked_scores = scores.masked_fill(mask == 0, -1) ### mask
# #     argsort = torch.argsort(masked_scores, dim=1, descending=True)
# #     ####################################################################################################################
# #     logs = []
# #     true_labels = []
# #     for idx in range(batch_size):
# #         pred_idxes_i = argsort[idx].tolist()
# #         top_k_labels_i = pred_idxes_i[:top_k]
# #         threhold_labels_i = pred_idxes_i[:top_k]
# #         # ==============================================================================================================
# #         for i in range(top_k, doc_num):
# #             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[top_k - 1]]:
# #                 threhold_labels_i.append(pred_idxes_i[i])
# #         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist()
# #         # ==============================================================================================================
# #         top_k_predictions.append(top_k_labels_i)
# #         threshold_predictions.append(threhold_labels_i)
# #         true_labels.append(labels_i)
# #         # ==============================================================================================================
# #         em_i, prec_i, recall_i, f1_i = sp_score(prediction=top_k_labels_i, gold=labels_i)
# #         t_em_i, t_prec_i, t_recall_i, t_f1_i = sp_score(prediction=threhold_labels_i, gold=labels_i)
# #         logs.append({
# #             'topk_sp_em': em_i,
# #             'topk_sp_f1': f1_i,
# #             'topk_sp_prec': prec_i,
# #             'topk_sp_recall': recall_i,
# #             'threshold_sp_em': t_em_i,
# #             'threshold_sp_f1': t_f1_i,
# #             'threshold_sp_prec': t_prec_i,
# #             'threshold_sp_recall': t_recall_i,
# #         })
# #         #################################################################################################################
# #         # Above is the support document prediction
# #         #################################################################################################################
# #         top_k_start_end_i, threshold_start_end_i = [], [] ## get the predicted document start, end index
# #         top_k_sent_i, threshold_sent_i = [], [] ## get the indexes of the absolute sent index
# #         for topk_pre_doc_idx in top_k_labels_i:
# #             doc_s_i = doc_start_pos[idx][topk_pre_doc_idx].data.item()
# #             doc_e_i = doc_end_pos[idx][topk_pre_doc_idx].data.item()
# #             top_k_start_end_i.append((doc_s_i, doc_e_i))
# #             #++++++++++++++++++++++++++++++++++++++++
# #             sent_idx_i = (sent2doc_map[idx] == topk_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
# #             if not isinstance(sent_idx_i, list):
# #                 sent_idx_i = [sent_idx_i]
# #             top_k_sent_i += sent_idx_i
# #
# #         for thresh_pre_doc_idx in threhold_labels_i:
# #             doc_s_i = doc_start_pos[idx][thresh_pre_doc_idx].data.item()
# #             doc_e_i = doc_end_pos[idx][thresh_pre_doc_idx].data.item()
# #             threshold_start_end_i.append((doc_s_i, doc_e_i))
# #             # ++++++++++++++++++++++++++++++++++++++++
# #             sent_idx_i = (sent2doc_map[idx] == thresh_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
# #             if not isinstance(sent_idx_i, list):
# #                 sent_idx_i = [sent_idx_i]
# #             threshold_sent_i += sent_idx_i
# #         ###############
# #         top_k_doc_start_end.append(top_k_start_end_i)
# #         threshold_doc_start_end.append(threshold_start_end_i)
# #         top_k_sent_idxes.append(top_k_sent_i)
# #         threshold_sent_idxes.append(threshold_sent_i)
# #
# #     res = {'log': logs,
# #            'top_k_doc': top_k_predictions,
# #            'threshold_doc': threshold_predictions,
# #            'true_doc': true_labels,
# #            'top_k_doc2token': top_k_doc_start_end,
# #            'top_k_sents': top_k_sent_idxes,
# #            'threshold_doc2token': threshold_doc_start_end,
# #            'threshold_sents': threshold_sent_idxes}
# #     return res
# # ########################################################################################################################
#
# # if max_pair_idx is None:
# #     for sent_idx in range(sent_number):
# #         if sent_mask[batch_idx][sent_idx] > 0:
# #             sent_start_i, sent_end_i = sent_start_positions[batch_idx][sent_idx], sent_end_positions[batch_idx][
# #                 sent_idx]
# #             sent_start_score_i = start_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
# #             sent_end_score_i = end_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
# #             print('start score {}\n {}\n{}'.format(sent_start_score_i,
# #                                                    start_scores[batch_idx][sent_start_i:(sent_end_i + 1)],
# #                                                    orig_start_score[batch_idx][sent_start_i:(sent_end_i + 1)]))
# #             print('end score {}\n{}\n{}'.format(sent_end_score_i,
# #                                                 end_scores[batch_idx][sent_start_i:(sent_end_i + 1)],
# #                                                 orig_end_score[batch_idx][sent_start_i:(sent_end_i + 1)]))
#
#
# import os
# import sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# import pandas as pd
# from pandas import DataFrame
# from multihopUtils.hotpotqaIOUtils import HOTPOT_DevData_Distractor
# from transformers import LongformerTokenizer
# from modelEvaluation.hotpot_evaluate_v1 import json_eval
# from torch import Tensor as T
# import torch
# import torch.nn.functional as F
# import swifter
# MAX_ANSWER_DECODE_LEN = 50
#
# ########################################################################################################################
# def sp_score(prediction, gold):
#     cur_sp_pred = set(prediction)
#     gold_sp_pred = set(gold)
#     tp, fp, fn = 0, 0, 0
#     for e in cur_sp_pred:
#         if e in gold_sp_pred:
#             tp += 1
#         else:
#             fp += 1
#     for e in gold_sp_pred:
#         if e not in cur_sp_pred:
#             fn += 1
#     prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
#     recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
#     f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
#     em = 1.0 if fp + fn == 0 else 0.0
#     return em, prec, recall, f1
# ########################################################################################################################
#
# def answer_type_prediction(type_scores: T, true_labels: T):
#     type_predicted_labels = torch.argmax(type_scores, dim=-1)
#     correct_num = (type_predicted_labels == true_labels).sum().data.item()
#     type_predicted_labels = type_predicted_labels.detach().tolist()
#     ans_type_map = {0: 'span', 1: 'yes', 2: 'no'}
#     type_predicted_labels = [ans_type_map[_] for _ in type_predicted_labels]
#     return correct_num, type_predicted_labels
#
# def answer_span_prediction(start_scores: T, end_scores: T, sent_start_positions: T, sent_end_positions: T, sent_mask: T):
#     batch_size, seq_len = start_scores.shape[0], start_scores.shape[1]
#     start_prob = torch.sigmoid(start_scores)
#     end_prob = torch.sigmoid(end_scores)
#     sent_number = sent_start_positions.shape[1]
#     if len(sent_start_positions.shape) > 1:
#         sent_start_positions = sent_start_positions.unsqueeze(dim=-1)
#     if len(sent_end_positions.shape) > 1:
#         sent_end_positions = sent_end_positions.unsqueeze(dim=-1)
#     answer_span_pairs = []
#     for batch_idx in range(batch_size):
#         max_score_i = 0
#         max_pair_idx = None
#         for sent_idx in range(sent_number):
#             if sent_mask[batch_idx][sent_idx] > 0:
#                 sent_start_i, sent_end_i = sent_start_positions[batch_idx][sent_idx], sent_end_positions[batch_idx][sent_idx]
#                 sent_start_score_i = start_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
#                 sent_end_score_i = end_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
#                 max_sent_core_i, start_idx, end_idx = answer_span_in_sentence(start_scores=sent_start_score_i, end_scores=sent_end_score_i)
#                 start_idx = start_idx + sent_start_i
#                 end_idx = end_idx + sent_end_i
#                 if max_score_i < max_sent_core_i:
#                     max_pair_idx = (start_idx, end_idx)
#         assert max_pair_idx is not None, 'max score {}'.format(max_score_i)
#         answer_span_pairs.append(max_pair_idx)
#     return answer_span_pairs
#
# def answer_span_in_sentence(start_scores: T, end_scores: T, max_ans_decode_len: int = MAX_ANSWER_DECODE_LEN):
#     sent_len = start_scores.shape[0]
#     score_matrix = torch.matmul(start_scores.view(1,-1).t(), end_scores.view(1,-1))
#     score_matrix = torch.triu(score_matrix)
#     if max_ans_decode_len < sent_len:
#         trip_len = sent_len - max_ans_decode_len
#         mask_upper_tri = torch.triu(torch.ones((trip_len, trip_len))).to(start_scores.device)
#         mask_upper_tri = F.pad(mask_upper_tri, [max_ans_decode_len, 0, 0, max_ans_decode_len])
#         score_matrix = score_matrix.masked_fill(mask_upper_tri==1, 0)
#     max_idx = torch.argmax(score_matrix)
#     start_idx, end_idx = max_idx // sent_len, max_idx % sent_len
#     start_idx, end_idx = start_idx.data.item(), end_idx.data.item()
#     score = score_matrix[start_idx][end_idx]
#     return score, start_idx, end_idx
#
# def add_id_context(data: DataFrame):
#     golden_data, _ = HOTPOT_DevData_Distractor()
#     data[['_id', 'context']] = golden_data[['_id', 'context']]
#     return data
#
# def convert2leadBoard(data: DataFrame, tokenizer: LongformerTokenizer):
#     ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     data = add_id_context(data=data)
#     ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     def process_row(row):
#         answer_type_prediction = row['aty_pred']
#         # support_doc_prediction = row['sd_pred']
#         ss_ds_pair = row['ss_ds_pair']
#         supp_sent_prediction_pair = ss_ds_pair
#         span_prediction = row['ans_span']
#         encode_ids = row['encode_ids']
#         context_docs = row['context']
#         if answer_type_prediction == 'span':
#             span_start, span_end = span_prediction[0], span_prediction[1]
#             answer_encode_ids = encode_ids[span_start:(span_end+1)]
#             answer_prediction = tokenizer.decode(answer_encode_ids, skip_special_tokens=True)
#             answer_prediction = answer_prediction.strip()
#             # print('pred {}\t true {}'.format(answer_prediction, row['answer']))
#         else:
#             answer_prediction = answer_type_prediction
#             # print('pred {}\t true {}'.format(answer_prediction, row['answer']))
#
#         # supp_doc_titles = [context_docs[idx][0] for idx in support_doc_prediction]
#         # return answer_prediction, supp_doc_titles, supp_title_sent_id
#         supp_title_sent_id = [(context_docs[x[0]][0], x[1]) for x in supp_sent_prediction_pair]
#         return answer_prediction, supp_title_sent_id
#
#     pred_names = ['answer', 'sp']
#     data[pred_names] = data.apply(lambda row: pd.Series(process_row(row)), axis=1)
#     res_names = ['_id', 'answer', 'sp']
#
#     predicted_data = data[res_names]
#     id_list = predicted_data['_id'].tolist()
#     answer_list = predicted_data['answer'].tolist()
#     sp_list = predicted_data['sp'].tolist()
#     answer_id_dict = dict(zip(id_list, answer_list))
#
#     sp_id_dict = dict(zip(id_list, sp_list))
#     predicted_data_dict = {'answer': answer_id_dict, 'sp': sp_id_dict}
#     golden_data, _ = HOTPOT_DevData_Distractor()
#     golden_data_dict = golden_data.to_dict(orient='records')
#     metrics = json_eval(prediction=predicted_data_dict, gold=golden_data_dict)
#     res_data_frame = pd.DataFrame.from_dict(predicted_data_dict)
#     return metrics, res_data_frame


# def score_label_pair(self, output_scores, sample):
    #     yn_scores = output_scores['yn_score']
    #     start_logits, end_logits = output_scores['span_score']
    #     #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #     # print(start_logits.shape, end_logits.shape, sample['ctx_attn_mask'].shape, sample['ctx_attn_mask'].sum(dim=1), sample['doc_lens'].sum(dim=1))
    #     answer_start_positions, answer_end_positions, yn_labels = sample['ans_start'], sample['ans_end'], sample['yes_no']
    #     if len(yn_labels.shape) > 0:
    #         yn_labels = yn_labels.squeeze(dim=-1)
    #     yn_num = (yn_labels > 0).sum().data.item()
    #     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #     supp_doc_scores, supp_head_tail_scores = output_scores['doc_score']
    #     supp_sent_scores = output_scores['sent_score']
    #     # ******************************************************************************************************************
    #     # ******************************************************************************************************************
    #     doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
    #     sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
    #     supp_head_position, supp_tail_position = sample['head_idx'], sample['tail_idx']
    #     doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
    #     sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
    #     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #     if len(answer_start_positions.size()) > 1:
    #         answer_start_positions = answer_start_positions.squeeze(-1)
    #     if len(answer_end_positions.size()) > 1:
    #         answer_end_positions = answer_end_positions.squeeze(-1)
    #     # sometimes the start/end positions are outside our reasonModel inputs, we ignore these terms
    #     ignored_index = start_logits.size(1)
    #     answer_start_positions.clamp_(0, ignored_index)
    #     answer_end_positions.clamp_(0, ignored_index)
    #     ##+++++++++++++++
    #     if yn_num > 0:
    #         ans_batch_idx = (yn_labels > 0).nonzero().squeeze()
    #         start_logits[ans_batch_idx] = -1
    #         end_logits[ans_batch_idx] = -1
    #         start_logits[ans_batch_idx, answer_start_positions[ans_batch_idx]] = 1
    #         end_logits[ans_batch_idx, answer_end_positions[ans_batch_idx]] = 1
    #     ##+++++++++++++++
    #     # ******************************************************************************************************************
    #     # ******************************************************************************************************************
    #     return {'yn': (yn_scores, yn_labels),
    #             'span': ((start_logits, end_logits), (answer_start_positions, answer_end_positions), ignored_index),
    #             'doc': (supp_doc_scores, doc_label, doc_mask),
    #             'doc_pair': (supp_head_tail_scores, supp_head_position, supp_tail_position),
    #             'sent': (supp_sent_scores, sent_label, sent_mask)}
    #
    # def loss_computation(self, output, sample):
    #     predict_label_pair = self.score_label_pair(output_scores=output, sample=sample)
    #     ##+++++++++++++
    #     yn_score, yn_label = predict_label_pair['yn']
    #     yn_loss_fct = MultiClassFocalLoss(num_class=3)
    #     yn_loss = yn_loss_fct.forward(yn_score, yn_label)
    #     ##+++++++++++++
    #     supp_loss_fct = PairwiseCEFocalLoss()
    #     supp_doc_scores, doc_label, doc_mask = predict_label_pair['doc']
    #     supp_doc_loss = supp_loss_fct.forward(scores=supp_doc_scores, targets=doc_label, target_len=doc_mask)
    #     ##+++++++++++++
    #     ##+++++++++++++
    #     supp_pair_doc_scores, head_position, tail_position = predict_label_pair['doc_pair']
    #     if supp_pair_doc_scores is None:
    #         supp_doc_pair_loss = torch.tensor(0.0).to(head_position.device)
    #     else:
    #         supp_pair_loss_fct = TriplePairwiseCEFocalLoss()
    #         supp_doc_pair_loss = supp_pair_loss_fct.forward(scores=supp_pair_doc_scores,
    #                                                         head_position=head_position,
    #                                                         tail_position=tail_position,
    #                                                         score_mask=doc_mask)
    #     ##+++++++++++++
    #     supp_sent_scores, sent_label, sent_mask = predict_label_pair['sent']
    #     supp_sent_loss = supp_loss_fct.forward(scores=supp_sent_scores, targets=sent_label, target_len=sent_mask)
    #     ##+++++++++++++
    #     span_logits, span_position, ignored_index = predict_label_pair['span']
    #     start_logits, end_logits = span_logits
    #     start_positions, end_positions = span_position
    #     # ++++++++++++++++++++++++++++++++++++++++++++++++
    #     span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
    #     start_loss = span_loss_fct(start_logits, start_positions)
    #     end_loss = span_loss_fct(end_logits, end_positions)
    #     # ++++++++++++++++++++++++++++++++++++++++++++++++
    #     span_loss = (start_loss + end_loss) / 2
    #     if span_loss > 20000:
    #         tokenizer = get_hotpotqa_longformer_tokenizer()
    #         ctx_encode_ids = sample['ctx_encode']
    #         print('start logits {}, start position {}'.format(start_logits, start_positions))
    #         print('end logits {}, end position {}'.format(end_logits, end_positions))
    #         batch_size = ctx_encode_ids.shape[0]
    #         for i in range(batch_size):
    #             start_i = start_positions[i]
    #             end_i = end_positions[i]
    #             print('decode answer={}'.format(tokenizer.decode(ctx_encode_ids[i][start_i:(end_i + 1)])))
    #             print('start logit {}, end logit {}'.format(start_logits[i][start_i], end_logits[i][end_i]))
    #     return {'yn_loss': yn_loss, 'span_loss': span_loss, 'doc_loss': supp_doc_loss, 'doc_pair_loss': supp_doc_pair_loss,
    #             'sent_loss': supp_sent_loss}