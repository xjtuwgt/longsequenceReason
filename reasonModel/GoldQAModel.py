from torch import Tensor as T
from torch import nn
import torch
from multihopUtils.longformerQAUtils import LongformerEncoder
from multihopUtils.hotpotQAlossUtils import MultiClassFocalLoss, PairwiseCEFocalLoss
from torch.nn import CrossEntropyLoss
from reasonModel.modelUtils import MLP

########################################################################################################################
########################################################################################################################
class LongformerHotPotQAModel(nn.Module):
    def __init__(self, longformer: LongformerEncoder, num_labels: int, fix_encoder=False):
        super().__init__()
        self.num_labels = num_labels
        self.longformer = longformer
        self.hidden_size = longformer.get_out_size()
        self.answer_type_outputs = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=3) ## yes, no, span question score
        self.qa_outputs = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=num_labels) ## span prediction score
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.sent_mlp = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1)
        ####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.mask_value = -1e9
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def get_representation(sub_model: LongformerEncoder, ids: T, attn_mask: T, global_attn_mask: T,
                           fix_encoder: bool = False) -> (
            T, T, T):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids,
                                                                                      attention_mask=attn_mask,
                                                                                      global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model.forward(input_ids=ids,
                                                                                  attention_mask=attn_mask,
                                                                                  global_attention_mask=global_attn_mask)
        return sequence_output, pooled_output, hidden_states

    def forward(self, sample):
        ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_encode'], sample['ctx_attn_mask'], sample['ctx_global_mask']
        sent_positions, special_marker = sample['sent_start'], sample['marker']
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sequence_output, _, _ = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        answer_type_scores = self.answer_type_prediction(sequence_output=sequence_output)
        start_logits, end_logits = self.span_prediction(sequence_output=sequence_output)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits = start_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        start_logits = start_logits.masked_fill(special_marker == 1, self.mask_value)
        end_logits = end_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        end_logits = end_logits.masked_fill(special_marker == 1, self.mask_value)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_scores = self.supp_sent_prediction(sequence_output=sequence_output, sent_position=sent_positions)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output = {'answer_type_score': answer_type_scores, 'answer_span_score': (start_logits, end_logits), 'sent_score': sent_scores}
        if self.training:
            loss_res = self.loss_computation(sample=sample, output_scores=output)
            return loss_res
        else:
            return output

    def answer_type_prediction(self, sequence_output: T):
        cls_emb = sequence_output[:, 0, :]
        scores = self.answer_type_outputs(cls_emb).squeeze(dim=-1)
        return scores

    def answer_span_prediction(self, sequence_output: T):
        logits = self.answer_span_prediction(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def supp_sent_prediction(self, sequence_output, sent_position):
        ##++++++++++++++++++++++++++++++++++++
        batch_size, sent_num = sent_position.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sent_num).to(sequence_output.device)
        sent_embed = sequence_output[batch_idx, sent_position]
        #####+++++++++++++++++++++++++++++++++
        sent_score = self.sent_mlp.forward(sent_embed).squeeze(dim=-1)
        return sent_score

    def answer_span_loss(self, start_logits: T, end_logits: T, start_positions: T, end_positions: T):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def answer_type_loss(self, answer_type_logits: T, true_labels: T):
        if len(true_labels.shape) > 1:
            true_lables = true_labels.squeeze(dim=-1)
        no_span_num = (true_lables > 0).sum().data.item()
        answer_type_loss_fct = MultiClassFocalLoss(num_class=3)
        yn_loss = answer_type_loss_fct.forward(answer_type_logits, true_lables)
        return yn_loss, no_span_num, true_lables

    def supp_sent_loss(self, sent_scores: T, sent_label: T, sent_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_sent_loss = supp_loss_fct.forward(scores=sent_scores, targets=sent_label, target_len=sent_mask)
        return supp_sent_loss

    def loss_computation(self, output_scores: dict, sample: dict):
        answer_type_scores = output_scores['answer_type_score']
        answer_type_labels = sample['yes_no']
        answer_type_loss_score, no_span_num, answer_type_labels = self.answer_type_loss(
            answer_type_logits=answer_type_scores,
            true_labels=answer_type_labels)
        ################################################################################################################
        answer_start_positions, answer_end_positions = sample['ans_start'], sample['ans_end']
        start_logits, end_logits = output_scores['answer_span_score']
        if no_span_num > 0:
            ans_batch_idx = (answer_type_labels > 0).nonzero().squeeze()
            start_logits[ans_batch_idx] = -10
            end_logits[ans_batch_idx] = -10
            start_logits[ans_batch_idx, answer_start_positions[ans_batch_idx]] = 10
            end_logits[ans_batch_idx, answer_end_positions[ans_batch_idx]] = 10
        ################################################################################################################
        answer_span_loss_score = self.answer_span_loss(start_logits=start_logits, end_logits=end_logits,
                                                       start_positions=answer_start_positions,
                                                       end_positions=answer_end_positions)
        ################################################################################################################
        sent_scores = output_scores['sent_score']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
        supp_sent_loss_score = self.supp_sent_loss(sent_scores=sent_scores, sent_label=sent_label, sent_mask=sent_mask)
        ################################################################################################################
        return {'answer_type_loss': answer_type_loss_score, 'span_loss': answer_span_loss_score, 'sent_loss': supp_sent_loss_score}