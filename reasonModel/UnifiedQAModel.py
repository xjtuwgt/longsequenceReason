from torch import Tensor as T
from torch import nn
import torch
from multihopUtils.longformerQAUtils import LongformerEncoder
from multihopUtils.hotpotQAlossUtils import MultiClassFocalLoss, PairwiseCEFocalLoss, TriplePairwiseCEFocalLoss
from reasonModel.Transformer import Transformer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_input, d_mid, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_input, d_mid)
        self.w_2 = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class BiLinear(nn.Module):
    def __init__(self, project_dim: int, args):
        super(BiLinear, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.bilinear_map = nn.Bilinear(in1_features=project_dim, in2_features=project_dim, out_features=1, bias=False)

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        doc_embed = self.inp_drop(doc_emb)
        scores = self.bilinear_map(doc_embed, q_embed).squeeze(dim=-1)
        return scores

class DotProduct(nn.Module):
    def __init__(self, args, transpose=False):
        super(DotProduct, self).__init__()
        self.inp_drop = nn.Dropout(args.input_drop)
        self.transpose = transpose

    def forward(self, query_emb: T, doc_emb: T):
        q_embed = self.inp_drop(query_emb)
        d_embed = self.inp_drop(doc_emb)
        if not self.transpose:
            scores = (q_embed * d_embed).sum(dim=-1)
        else:
            scores = torch.matmul(q_embed, d_embed.transpose(-1,-2))
        return scores

def compute_smooth_sigmoid(scores: T, smooth_factor=1e-7):
    prob = torch.sigmoid(scores)
    prob = torch.clamp(prob, smooth_factor, 1.0 - smooth_factor)
    return prob

def compute_smooth_reverse_sigmoid(prob: T):
    scores = torch.log(prob/(1.0 - prob))
    return scores

########################################################################################################################
########################################################################################################################
class LongformerHotPotQAModel(nn.Module):
    def __init__(self, longformer: LongformerEncoder, num_labels: int, args, fix_encoder=False):
        super().__init__()
        self.num_labels = num_labels
        self.longformer = longformer
        self.hidden_size = longformer.get_out_size()
        self.yn_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=3) ## yes, no, span question score
        self.qa_outputs = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=num_labels) ## span prediction score
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.score_model_name = args.score_model_name ## supp doc score/supp sent score
        self.hop_model_name = args.hop_model_name ## triple score
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.with_graph = args.with_graph == 1
        if self.with_graph:
            self.transformer_layer = Transformer(d_model=self.hidden_size, heads=args.heads)
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_training = args.with_graph_training == 1
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.score_model_name not in ['MLP']:
            raise ValueError('reasonModel %s not supported' % self.score_model_name)
        else:
            self.doc_mlp = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) if self.score_model_name == 'MLP' else None
            self.sent_mlp = PositionwiseFeedForward(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) if self.score_model_name == 'MLP' else None

        if self.hop_model_name not in ['DotProduct', 'BiLinear']:
            self.hop_model_name = None
        else:
            self.hop_doc_dotproduct = DotProduct(args=args) if self.hop_model_name == 'DotProduct' else None
            self.hop_doc_bilinear = BiLinear(args=args, project_dim=self.hidden_size) if self.hop_model_name == 'BiLinear' else None

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
        doc_positions, sent_positions = sample['doc_start'], sample['sent_start']
        special_marker = sample['marker']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (self.hop_model_name is not None) and self.training:
            head_doc_positions, tail_doc_positions = sample['head_idx'], sample['tail_idx']
            head_tail_pair = (head_doc_positions, tail_doc_positions)
        else:
            head_tail_pair = None
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sequence_output, _, _ = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        answer_type_scores = self.answer_type_prediction(sequence_output=sequence_output)
        start_logits, end_logits = self.answer_span_prediction(sequence_output=sequence_output)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_sent_mask, doc_sent_mask = sample['ss_mask'], sample['sd_mask']
        sent_scores, doc_scores, doc_pair_scores = self.supp_doc_sent_prediction(sequence_output=sequence_output,
                                                                                                 doc_position=doc_positions,
                                                                                                 sent_position=sent_positions,
                                                                                                 sent_sent_mask=sent_sent_mask,
                                                                                                 doc_sent_mask=doc_sent_mask,
                                                                                                 head_tail_pair=head_tail_pair)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits = start_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        start_logits = start_logits.masked_fill(special_marker == 1, self.mask_value)
        end_logits = end_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        end_logits = end_logits.masked_fill(special_marker == 1, self.mask_value)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output = {'answer_type_score': answer_type_scores, 'answer_span_score': (start_logits, end_logits),
                  'doc_score': (doc_scores, doc_pair_scores), 'sent_score': sent_scores}
        if self.training:
            loss_res = self.multi_loss_computation(sample=sample, output_scores=output)
            return loss_res
        else:
            assert doc_pair_scores == None
            return output

    def hierartical_score(self, span_start_score: T, span_end_score: T, doc_scores: T, sent_scores: T,
                          doc_lens: T, sent_lens: T, doc2sent_map: T, sent2token_map: T):
        doc_scores = doc_scores.masked_fill(doc_lens==0, self.mask_value)
        doc_weights = F.softmax(doc_scores, dim=-1)
        doc2sent_weights = doc_weights.gather(1, doc2sent_map)
        sent2token_weights = doc2sent_weights.gather(1, sent2token_map)

        sent_prob = compute_smooth_sigmoid(scores=sent_scores)
        span_start_prob = compute_smooth_sigmoid(scores=span_start_score)
        span_end_prob = compute_smooth_sigmoid(scores=span_end_score)

        sent_prob = sent_prob * doc2sent_weights
        span_start_prob = span_start_prob * sent2token_weights
        span_end_prob = span_end_prob * sent2token_weights

        sent_logit = compute_smooth_reverse_sigmoid(prob=sent_prob)
        span_start_logit = compute_smooth_reverse_sigmoid(prob=span_start_prob)
        span_end_logit = compute_smooth_reverse_sigmoid(prob=span_end_prob)
        return span_start_logit, span_end_logit, sent_logit

    def answer_type_prediction(self, sequence_output: T):
        cls_emb = sequence_output[:, 0, :]
        scores = self.yn_outputs(cls_emb).squeeze(dim=-1)
        return scores

    def answer_span_prediction(self, sequence_output: T):
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def supp_doc_sent_prediction(self, sequence_output, doc_position, doc_sent_mask, sent_position, sent_sent_mask, head_tail_pair=None):
        ##++++++++++++++++++++++++++++++++++++
        batch_size, sent_num = sent_position.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, sent_num).to(sequence_output.device)
        sent_embed = sequence_output[batch_idx, sent_position]
        # if self.with_graph:
        #     sent_embed = self.transformer_layer.forward(query=sent_embed, key=sent_embed, value=sent_embed, x_mask=sent_sent_mask)
        #####++++++++++++++++++++
        sent_model_func = {'MLP': self.MLP}
        if self.score_model_name in sent_model_func:
            sent_score = sent_model_func[self.score_model_name](sent_embed, mode='sentence').squeeze(dim=-1)
        else:
            raise ValueError('Score Model %s not supported' % self.score_model_name)
        ####+++++++++++++++++++++
        ##++++++++++++++++++++++++++++++++++++
        batch_size, doc_num = doc_position.shape
        batch_idx = torch.arange(0, batch_size).view(batch_size, 1).repeat(1, doc_num).to(sequence_output.device)
        doc_embed = sequence_output[batch_idx, doc_position]
        if self.with_graph:
            # ##+++++++++++++++
            # sent_embed = self.transformer_layer.forward(query=sent_embed, key=sent_embed, value=sent_embed,
            #                                             x_mask=sent_sent_mask)
            # ##+++++++++++++++
            doc_embed = self.transformer_layer.forward(query=doc_embed, key=sent_embed, value=sent_embed, x_mask=doc_sent_mask)
        ##++++++++++++++++++++++++++++++++++++
        #####++++++++++++++++++++
        doc_model_func = {'MLP': self.MLP}
        if self.score_model_name in doc_model_func:
            doc_score = doc_model_func[self.score_model_name](doc_embed, mode='document').squeeze(dim=-1)
        else:
            raise ValueError('Score Model %s not supported' % self.score_model_name)
        #####++++++++++++++++++++
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_pair_score = None
        if head_tail_pair is not None:
            head_position, tail_position = head_tail_pair
            # ######++++++++++++++++++++++++++++++++++++++
            query_emb = sequence_output[:, 1, :]
            query_emb = query_emb.unsqueeze(dim=1).repeat([1, doc_num, 1])
            # # ######++++++++++++++++++++++++++++++++++++++
            if len(head_position.shape) > 1:
                head_position = head_position.squeeze(dim=-1)
            p_batch_idx = torch.arange(0, batch_size).to(sequence_output.device)
            head_emb = doc_embed[p_batch_idx, head_position].unsqueeze(dim=1).repeat([1, doc_num, 1])
            ###################
            head_emb = head_emb * query_emb
            ###################
            hop_model_func = {'DotProduct': self.Hop_DotProduct, 'BiLinear': self.Hop_BiLinear}
            if self.hop_model_name in hop_model_func:
                doc_pair_score = hop_model_func[self.hop_model_name](head_emb, doc_embed).squeeze(dim=-1)
            else:
                raise ValueError('Hop score mode %s not supported' % self.hop_model_name)
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return sent_score, doc_score, doc_pair_score

    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def Hop_DotProduct(self, head_emb: T, tail_emb: T) -> T:
        score = self.hop_doc_dotproduct.forward(head_emb, tail_emb)
        return score

    def Hop_BiLinear(self, head_emb: T, tail_emb: T) -> T:
        score = self.hop_doc_bilinear.forward(head_emb, tail_emb)
        return score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def MLP(self, ctx_emb: T, mode: str) -> T:
        if mode == 'document':
            query_ctx_emb = ctx_emb
            score = self.doc_mlp.forward(query_ctx_emb)
        elif mode == 'sentence':
            query_ctx_emb = ctx_emb
            score = self.sent_mlp.forward(query_ctx_emb)
        else:
            raise ValueError('mode %s not supported' % mode)
        return score

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

    def supp_doc_loss(self, doc_scores: T, doc_label: T, doc_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_doc_loss = supp_loss_fct.forward(scores=doc_scores, targets=doc_label, target_len=doc_mask)
        return supp_doc_loss

    def doc_hop_loss(self, doc_pair_scores: T, head_position: T, tail_position: T, doc_mask: T):
        supp_pair_loss_fct = TriplePairwiseCEFocalLoss()
        supp_doc_pair_loss = supp_pair_loss_fct.forward(scores=doc_pair_scores,
                                                        head_position=head_position,
                                                        tail_position=tail_position,
                                                        score_mask=doc_mask)
        return supp_doc_pair_loss

    def supp_sent_loss(self, sent_scores: T, sent_label: T, sent_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_sent_loss = supp_loss_fct.forward(scores=sent_scores, targets=sent_label, target_len=sent_mask)
        return supp_sent_loss

    def multi_loss_computation(self, output_scores: dict, sample: dict):
        answer_type_scores = output_scores['answer_type_score']
        answer_type_labels = sample['yes_no']
        answer_type_loss_score, no_span_num, answer_type_labels = self.answer_type_loss(answer_type_logits=answer_type_scores,
                                                                                  true_labels=answer_type_labels)
        #######################################################################
        answer_start_positions, answer_end_positions = sample['ans_start'], sample['ans_end']
        start_logits, end_logits = output_scores['answer_span_score']
        if no_span_num > 0:
            ans_batch_idx = (answer_type_labels > 0).nonzero().squeeze()
            start_logits[ans_batch_idx] = -1
            end_logits[ans_batch_idx] = -1
            start_logits[ans_batch_idx, answer_start_positions[ans_batch_idx]] = 1
            end_logits[ans_batch_idx, answer_end_positions[ans_batch_idx]] = 1
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_span_loss_score = self.answer_span_loss(start_logits=start_logits, end_logits=end_logits,
                                                 start_positions=answer_start_positions, end_positions=answer_end_positions)
        #######################################################################
        doc_scores, doc_pair_scores = output_scores['doc_score']
        doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
        doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
        supp_doc_loss_score = self.supp_doc_loss(doc_scores=doc_scores, doc_label=doc_label, doc_mask=doc_mask)
        if doc_pair_scores is not None:
            supp_head_position, supp_tail_position = sample['head_idx'], sample['tail_idx']
            supp_doc_pair_loss_score = self.doc_hop_loss(doc_pair_scores=doc_pair_scores, head_position=supp_head_position,
                                                   tail_position=supp_tail_position, doc_mask=doc_mask)
        else:
            supp_doc_pair_loss_score = torch.tensor(0.0).to(doc_label.device)
        #######################################################################
        sent_scores = output_scores['sent_score']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
        supp_sent_loss_score = self.supp_sent_loss(sent_scores=sent_scores, sent_label=sent_label, sent_mask=sent_mask)

        return {'yn_loss': answer_type_loss_score, 'span_loss': answer_span_loss_score,
                'doc_loss': supp_doc_loss_score, 'doc_pair_loss': supp_doc_pair_loss_score,
                'sent_loss': supp_sent_loss_score}