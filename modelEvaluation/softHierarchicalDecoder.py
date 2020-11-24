import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import logging
from time import time
import torch
from pandas import DataFrame
from torch import Tensor as T
from modelTrain.modelTrainUtils import MASK_VALUE
from modelEvaluation.hotpotEvaluationUtils import sp_score
########################################################################################################################
########################################################################################################################
def soft_hierartical_decoder(model, device, test_data_loader, doc_topk, args):
    '''
            Evaluate the reasonModel on test or valid datasets
    '''
    model.eval()
    ###########################################################
    start_time = time()
    test_dataset = test_data_loader
    doc_logs, topk_sent_logs, thresh_sent_logs = [], [], []
    step = 0
    N = 0
    total_steps = len(test_dataset)
    # **********************************************************
    topk_support_doc_pred_results, threshold_support_doc_pred_results, answer_type_pred_results = [], [], []
    topk_support_sent_pred_results, topk_support_sent_doc_sent_pair_results = [], []
    thresh_support_sent_pred_results, thresh_support_sent_doc_sent_pair_results = [], []
    topk_answer_span_results, thresh_answer_span_results = [], []
    correct_answer_num = 0
    encode_id_results = []
    # **********************************************************
    with torch.no_grad():
        for test_sample in test_dataset:
            if args.cuda:
                sample = dict()
                for key, value in test_sample.items():
                    sample[key] = value.to(device)
            else:
                sample = test_sample
            output = model(sample)
            N = N + sample['doc_labels'].shape[0]
    #         eval_res = hierartical_metric_computation(output_scores=output, sample=sample, doc_topk=doc_topk, args=args)
    #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         correct_yn, yn_predicted_labels = eval_res['answer_type_pred']
    #         correct_answer_num = correct_answer_num + correct_yn
    #         answer_type_pred_results = answer_type_pred_results  + yn_predicted_labels
    #         # **********************************************************************************************************
    #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         doc_res_dict_i = eval_res['supp_doc_pred']
    #         doc_logs += doc_res_dict_i['log']
    #         topk_support_doc_pred_results += doc_res_dict_i['top_k_doc']
    #         threshold_support_doc_pred_results += doc_res_dict_i['threshold_doc']
    #         # **********************************************************************************************************
    #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         topk_sent_res_dict_i = eval_res['topk_sent_pred']
    #         topk_sent_logs += topk_sent_res_dict_i['log']
    #         topk_support_sent_pred_results += topk_sent_res_dict_i['prediction']
    #         topk_support_sent_doc_sent_pair_results += topk_sent_res_dict_i['doc_sent_pair']
    #         # **********************************************************************************************************
    #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         thresh_sent_res_dict_i = eval_res['threshold_sent_pred']
    #         thresh_sent_logs += thresh_sent_res_dict_i['log']
    #         thresh_support_sent_pred_results += thresh_sent_res_dict_i['prediction']
    #         thresh_support_sent_doc_sent_pair_results += thresh_sent_res_dict_i['doc_sent_pair']
    #         # **********************************************************************************************************
    #         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         topk_answer_span_results += eval_res['topk_span_pred']
    #         thresh_answer_span_results += eval_res['threshold_span_pred']
    #         encode_i = sample['ctx_encode'].detach().tolist()
    #         encode_id_results += encode_i
    #         # **********************************************************************************************************
    #         step += 1
    #         if step % args.test_log_steps == 0:
    #             logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(step, total_steps, time()-start_time))
    #
    # ###############################################################
    # doc_metrics, topk_sent_metrics, thresh_sent_metrics = {}, {}, {}
    # for metric in doc_logs[0].keys():
    #     doc_metrics[metric] = sum([log[metric] for log in doc_logs]) / len(doc_logs)
    # for metric in topk_sent_logs[0].keys():
    #     topk_sent_metrics[metric] = sum([log[metric] for log in topk_sent_logs]) / len(topk_sent_logs)
    # for metric in thresh_sent_logs[0].keys():
    #     thresh_sent_metrics[metric] = sum([log[metric] for log in thresh_sent_logs]) / len(thresh_sent_logs)
    # ##=================================================
    # answer_type_accuracy = '{:.4f}'.format(correct_answer_num * 1.0/N)
    # result_dict = {'aty_pred': answer_type_pred_results,
    #                'topk_sd_pred': topk_support_doc_pred_results,
    #                'thresh_sd_pred': threshold_support_doc_pred_results,
    #                'topk_ss_pred': topk_support_sent_pred_results,
    #                'thresh_ss_pred': threshold_support_doc_pred_results,
    #                'topk_ds_pair': topk_support_sent_doc_sent_pair_results,
    #                'thresh_ds_pair': thresh_support_sent_doc_sent_pair_results,
    #                'topk_ans_span': topk_answer_span_results,
    #                'thresh_ans_span': thresh_answer_span_results,
    #                'encode_ids': encode_id_results}  ## for detailed results checking
    # res_data_frame = DataFrame(result_dict)
    # # ##=================================================
    # return {'supp_doc_metrics': doc_metrics, 'topk_supp_sent_metrics': topk_sent_metrics,
    #         'thresh_supp_sent_metrics': thresh_sent_metrics,
    #         'answer_type_acc': answer_type_accuracy, 'res_dataframe': res_data_frame}


# def hierartical_metric_computation(output_scores: dict, sample: dict, doc_topk, args):
#     #'yn_score', 'span_score', 'doc_score': (supp_doc_scores, None), 'sent_score':
#     yn_scores = output_scores['yn_score']
#     yn_true_labels = sample['yes_no']
#     if len(yn_true_labels.shape) > 1:
#         yn_true_labels = yn_true_labels.squeeze(dim=-1)
#     yn_predicted_labels = torch.argmax(yn_scores, dim=-1)
#     correct_yn = (yn_predicted_labels == yn_true_labels).sum().data.item()
#     yn_predicted_labels = yn_predicted_labels.detach().tolist()
#     ####################################################################################################################
#     doc_scores, temp = output_scores['doc_score']
#     assert temp is None
#     doc_mask = sample['doc_lens']
#     true_doc_labels = sample['doc_labels']
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     doc_start_position, doc_end_position = sample['doc_start'], sample['doc_end'] ## doc start and end position for answer span prediction
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     sent2doc_map = sample['s2d_map'] ## absolute sentence index map to document index, e.g., 150 sentence indexes to 10 documents
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     doc_res_dict = hierartical_supp_doc_prediction(doc_scores=doc_scores, mask=doc_mask,
#                                        labels=true_doc_labels, doc_start_pos=doc_start_position,
#                                        doc_end_pos=doc_end_position, sent2doc_map=sent2doc_map, top_k=doc_topk, threshold=args.doc_threshold)
#     ####################################################################################################################
#     ####################################################################################################################
#     ####################################################################################################################
#     sent_scores = output_scores['sent_score']
#     sentIndoc_map = sample['sInd_map'] ## absolute sentence index map to relative sentence index, e.g., 150 sentence indexes to 10 documents
#     topk_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['top_k_sents'])
#     threshold_sent_scores = sent_score_extraction(sent_scores=sent_scores, doc2sent_idexes=doc_res_dict['threshold_sents'])
#     true_sent_labels = sample['sent_labels']
#     sent_lens = sample['sent_lens']
#     topk_sent_res_dict = supp_sent_predictions(scores=topk_sent_scores, labels=true_sent_labels,
#                                                mask=sent_lens, sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map,
#                                                pred_num=2, threshold=args.sent_threshold)
#     threshold_sent_res_dict = supp_sent_predictions(scores=threshold_sent_scores, labels=true_sent_labels, mask=sent_lens,
#                                                     sent2doc_map=sent2doc_map, sentIndoc_map=sentIndoc_map, pred_num=2,
#                                                     threshold=args.sent_threshold)
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     span_start_scores, span_end_scores = output_scores['span_score']
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     topk_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
#     topk_span_start_i = torch.argmax(topk_span_start_scores, dim=-1)
#     topk_span_start_i = topk_span_start_i.detach().tolist()
#
#     topk_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['top_k_doc2token'])
#     topk_span_end_i = torch.argmax(topk_span_end_scores, dim=-1)
#     topk_span_end_i = topk_span_end_i.detach().tolist()
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     threshold_span_start_scores = token_score_extraction(token_scores=span_start_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
#     threshold_span_start_i = torch.argmax(threshold_span_start_scores, dim=-1)
#     threshold_span_start_i = threshold_span_start_i.detach().tolist()
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     threshold_span_end_scores = token_score_extraction(token_scores=span_end_scores, doc_start_end_pair_list=doc_res_dict['threshold_doc2token'])
#     threshold_span_end_i = torch.argmax(threshold_span_end_scores, dim=-1)
#     threshold_span_end_i = threshold_span_end_i.detach().tolist()
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     topk_span_start_end_pair = list(zip(topk_span_start_i, topk_span_end_i))
#     threshold_span_start_end_pair = list(zip(threshold_span_start_i, threshold_span_end_i))
#     # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     res = {'answer_type_pred': (correct_yn, yn_predicted_labels),
#            'supp_doc_pred': doc_res_dict,
#            'topk_sent_pred': topk_sent_res_dict,
#            'threshold_sent_pred': threshold_sent_res_dict,
#            'topk_span_pred': topk_span_start_end_pair,
#            'threshold_span_pred': threshold_span_start_end_pair}
#     return res
#
# def sent_score_extraction(sent_scores: T, doc2sent_idexes: list):
#     batch_size, sent_num = sent_scores.shape
#     temp_scores = torch.empty(batch_size, sent_num).fill_(MASK_VALUE).to(sent_scores.device)
#     for idx in range(batch_size):
#         temp_scores[idx, doc2sent_idexes[idx]] = sent_scores[idx, doc2sent_idexes[idx]]
#     return temp_scores
#
# def token_score_extraction(token_scores: T, doc_start_end_pair_list: list):
#     batch_size, seq_num = token_scores.shape
#     temp_scores = torch.empty(batch_size, seq_num).fill_(MASK_VALUE).to(token_scores.device)
#     for idx in range(batch_size):
#         doc_start_end_i = doc_start_end_pair_list[idx]
#         for dox_idx, doc_pair in enumerate(doc_start_end_i):
#             start_i, end_i = doc_pair
#             temp_scores[idx][start_i:(end_i+1)] = token_scores[idx][start_i:(end_i+1)]
#     return temp_scores
#
# def supp_sent_predictions(scores: T, labels: T, mask: T, sent2doc_map: T, sentIndoc_map: T, pred_num=2, threshold=0.9):
#     batch_size, sample_size = scores.shape[0], scores.shape[1]
#     scores = torch.sigmoid(scores)
#     masked_scores = scores.masked_fill(mask == 0, -1)
#     argsort = torch.argsort(masked_scores, dim=1, descending=True)
#     logs = []
#     predicted_labels = []
#     # true_labels = []
#     doc_sent_pair_list = []
#     for idx in range(batch_size):
#         # ============================================================================================================
#         doc_fact_i = sent2doc_map[idx].detach().tolist()
#         sent_fact_i = sentIndoc_map[idx].detach().tolist()
#         doc_sent_pair_i = list(zip(doc_fact_i, sent_fact_i)) ## pair of (doc_id, sent_id) --> number of pairs = number of all sentences in long sequence
#         doc_sent_pair_list.append(doc_sent_pair_i) ## for final leadboard evaluation
#         # ============================================================================================================
#         pred_idxes_i = argsort[idx].tolist()
#         pred_labels_i = pred_idxes_i[:pred_num]
#         for i in range(pred_num, sample_size):
#             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[pred_num-1]]:
#                 pred_labels_i.append(pred_idxes_i[i])
#         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist() ## sentence labels: [0, 1, 2], support doc: [0, 1]. 1 and 2 are support sentences
#         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         predicted_labels.append(pred_labels_i)
#         # true_labels.append(labels_i)
#         # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_labels_i, gold=labels_i)
#         logs.append({
#             'sp_em': em_i,
#             'sp_f1': f1_i,
#             'sp_prec': prec_i,
#             'sp_recall':recall_i
#         })
#     res = {'log': logs, 'prediction': predicted_labels, 'doc_sent_pair': doc_sent_pair_list}
#     return res
#
#
# def hierartical_supp_doc_prediction(doc_scores: T, labels: T, mask: T, doc_start_pos: T, doc_end_pos: T, sent2doc_map, top_k=2, threshold=0.9):
#     batch_size, doc_num = doc_scores.shape
#     top_k_predictions = []
#     threshold_predictions = []
#     ####################################################################################################################
#     top_k_doc_start_end = [] ### for answer span prediction
#     top_k_sent_idxes = [] ## for support sentence prediction
#     ####################################################################################################################
#     threshold_doc_start_end = [] ### for answer span prediction
#     threshold_sent_idxes = [] ### for support sentence prediction
#     ####################################################################################################################
#     scores = torch.sigmoid(doc_scores)
#     masked_scores = scores.masked_fill(mask == 0, -1) ### mask
#     argsort = torch.argsort(masked_scores, dim=1, descending=True)
#     ####################################################################################################################
#     logs = []
#     true_labels = []
#     for idx in range(batch_size):
#         pred_idxes_i = argsort[idx].tolist()
#         top_k_labels_i = pred_idxes_i[:top_k]
#         threhold_labels_i = pred_idxes_i[:top_k]
#         # ==============================================================================================================
#         for i in range(top_k, doc_num):
#             if masked_scores[idx, pred_idxes_i[i]] > threshold * masked_scores[idx, pred_idxes_i[top_k - 1]]:
#                 threhold_labels_i.append(pred_idxes_i[i])
#         labels_i = (labels[idx] > 0).nonzero(as_tuple=False).squeeze().tolist()
#         # ==============================================================================================================
#         top_k_predictions.append(top_k_labels_i)
#         threshold_predictions.append(threhold_labels_i)
#         true_labels.append(labels_i)
#         # ==============================================================================================================
#         em_i, prec_i, recall_i, f1_i = sp_score(prediction=top_k_labels_i, gold=labels_i)
#         t_em_i, t_prec_i, t_recall_i, t_f1_i = sp_score(prediction=threhold_labels_i, gold=labels_i)
#         logs.append({
#             'topk_sp_em': em_i,
#             'topk_sp_f1': f1_i,
#             'topk_sp_prec': prec_i,
#             'topk_sp_recall': recall_i,
#             'threshold_sp_em': t_em_i,
#             'threshold_sp_f1': t_f1_i,
#             'threshold_sp_prec': t_prec_i,
#             'threshold_sp_recall': t_recall_i,
#         })
#         #################################################################################################################
#         # Above is the support document prediction
#         #################################################################################################################
#         top_k_start_end_i, threshold_start_end_i = [], [] ## get the predicted document start, end index
#         top_k_sent_i, threshold_sent_i = [], [] ## get the indexes of the absolute sent index
#         for topk_pre_doc_idx in top_k_labels_i:
#             doc_s_i = doc_start_pos[idx][topk_pre_doc_idx].data.item()
#             doc_e_i = doc_end_pos[idx][topk_pre_doc_idx].data.item()
#             top_k_start_end_i.append((doc_s_i, doc_e_i))
#             #++++++++++++++++++++++++++++++++++++++++
#             sent_idx_i = (sent2doc_map[idx] == topk_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
#             if not isinstance(sent_idx_i, list):
#                 sent_idx_i = [sent_idx_i]
#             top_k_sent_i += sent_idx_i
#
#         for thresh_pre_doc_idx in threhold_labels_i:
#             doc_s_i = doc_start_pos[idx][thresh_pre_doc_idx].data.item()
#             doc_e_i = doc_end_pos[idx][thresh_pre_doc_idx].data.item()
#             threshold_start_end_i.append((doc_s_i, doc_e_i))
#             # ++++++++++++++++++++++++++++++++++++++++
#             sent_idx_i = (sent2doc_map[idx] == thresh_pre_doc_idx).nonzero(as_tuple=False).squeeze().tolist()
#             if not isinstance(sent_idx_i, list):
#                 sent_idx_i = [sent_idx_i]
#             threshold_sent_i += sent_idx_i
#         ###############
#         top_k_doc_start_end.append(top_k_start_end_i)
#         threshold_doc_start_end.append(threshold_start_end_i)
#         top_k_sent_idxes.append(top_k_sent_i)
#         threshold_sent_idxes.append(threshold_sent_i)
#
#     res = {'log': logs,
#            'top_k_doc': top_k_predictions,
#            'threshold_doc': threshold_predictions,
#            'true_doc': true_labels,
#            'top_k_doc2token': top_k_doc_start_end,
#            'top_k_sents': top_k_sent_idxes,
#            'threshold_doc2token': threshold_doc_start_end,
#            'threshold_sents': threshold_sent_idxes}
#     return res
########################################################################################################################

def answer_span_prediction(start_scores: T, end_scores: T):
    return