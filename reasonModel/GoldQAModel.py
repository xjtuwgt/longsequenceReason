from torch import Tensor as T
from torch import nn
import torch
from multihopUtils.longformerQAUtils import LongformerEncoder
from multihopUtils.hotpotQAlossUtils import MultiClassFocalLoss, PairwiseCEFocalLoss
from torch.nn import CrossEntropyLoss
from reasonModel.Transformer import PositionwiseFeedForward

########################################################################################################################
########################################################################################################################
class LongformerHotPotQAModel(nn.Module):
    def __init__(self, longformer: LongformerEncoder, num_labels: int, fix_encoder=False):
        super().__init__()
        self.num_labels = num_labels
        self.longformer = longformer
        self.hidden_size = longformer.get_out_size()
        self.yn_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=3) ## yes, no, span question score
        self.qa_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=num_labels) ## span prediction score
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.sent_mlp = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1)
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        sent_positions = sample['sent_start']
        special_marker = sample['marker']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sequence_output, _, _ = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        yn_scores = self.yes_no_prediction(sequence_output=sequence_output)
        start_logits, end_logits = self.span_prediction(sequence_output=sequence_output)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits = start_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        start_logits = start_logits.masked_fill(special_marker == 1, self.mask_value)
        end_logits = end_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        end_logits = end_logits.masked_fill(special_marker == 1, self.mask_value)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_scores = self.supp_sent_prediction(sequence_output=sequence_output, sent_position=sent_positions)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output = {'yn_score': yn_scores, 'span_score': (start_logits, end_logits), 'sent_score': sent_scores}
        if self.training:
            loss_res = self.loss_computation(sample=sample, output_scores=output)
            return loss_res
        else:
            return output

    def yes_no_prediction(self, sequence_output: T):
        cls_emb = sequence_output[:, 0, :]
        scores = self.yn_outputs(cls_emb).squeeze(dim=-1)
        return scores

    def span_prediction(self, sequence_output: T):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def supp_sent_prediction(self, sequence_output, sent_position):
        ##++++++++++++++++++++++++++++++++++++
        batch_size, sent_num = sent_position.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sent_num).to(sequence_output.device)
        sent_embed = sequence_output[batch_idx, sent_position]
        #####++++++++++++++++++++
        sent_score = self.sent_mlp.forward(sent_embed).squeeze(dim=-1)
        return sent_score

    def loss_computation(self, output_scores, sample):
        yn_score = output_scores['yn_score']
        yn_label = sample['yes_no']
        if len(yn_label.shape) > 1:
            yn_label = yn_label.squeeze(dim=-1)
        yn_loss_fct = MultiClassFocalLoss(num_class=3)
        yn_loss = yn_loss_fct.forward(yn_score, yn_label)
        yn_num = (yn_label > 0).sum().data.item()
        ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_score = output_scores['sent_score']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)

        sent_score = sent_score.masked_fill(sent_lens == 0, self.mask_value)
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_sent_loss = supp_loss_fct.forward(scores=sent_score, targets=sent_label, target_len=sent_mask)
        ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits, end_logits = output_scores['span_score']
        answer_start_positions, answer_end_positions = sample['ans_start'], sample['ans_end']

        if len(answer_start_positions.size()) > 1:
            answer_start_positions = answer_start_positions.squeeze(-1)
        if len(answer_end_positions.size()) > 1:
            answer_end_positions = answer_end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our reasonModel inputs, we ignore these terms
        #########################
        if yn_num > 0:
            ans_batch_idx = (yn_label > 0).nonzero().squeeze()
            start_logits[ans_batch_idx] = -10
            end_logits[ans_batch_idx] = -10
            start_logits[ans_batch_idx, answer_start_positions[ans_batch_idx]] = 10
            end_logits[ans_batch_idx, answer_end_positions[ans_batch_idx]] = 10
        #########################
        ignored_index = start_logits.size(1)
        answer_start_positions.clamp_(0, ignored_index)
        answer_end_positions.clamp_(0, ignored_index)
        span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = span_loss_fct(start_logits, answer_start_positions)
        end_loss = span_loss_fct(end_logits, answer_end_positions)
        span_loss = (start_loss + end_loss) / 2
        return {'yn_loss': yn_loss, 'span_loss': span_loss, 'sent_loss': supp_sent_loss}