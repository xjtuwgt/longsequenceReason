from transformers import LongformerModel, LongformerTokenizer
from torch import Tensor as T
from torch import nn
import torch
import logging
from transformers.configuration_longformer import LongformerConfig
PRE_TAINED_LONFORMER_BASE = 'allenai/longformer-base-4096'

class QATensorizer(object):
    """
    Component for all text to reasonModel input data conversions and related utility methods
    """
    def text_to_tensor(self, text: str, add_special_tokens: bool = True):
        raise NotImplementedError

    def text_encode(self, text: str, add_special_tokens: bool = True):
        raise NotImplementedError

    def token_ids_padding(self, token_ids):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def token_ids_to_tensor(self, token_ids):
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, token_ids_tensor: T):
        raise NotImplementedError

    def get_global_attn_mask(self, token_ids_tensor: T, gobal_mask_idxs):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

class LongformerQATensorizer(QATensorizer):
    def __init__(self, tokenizer: LongformerTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, add_special_tokens: bool = True):
        text_tokens = self.tokenizer.tokenize(text=text)
        token_ids = self.tokenizer.encode(text_tokens, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def text_encode(self, text: str, add_special_tokens: bool = True):
        text_tokens = self.tokenizer.tokenize(text=text)
        token_ids = self.tokenizer.encode(text_tokens, add_special_tokens=add_special_tokens,
                                              pad_to_max_length=False, truncation=True)
        return token_ids

    def token_ids_padding(self, token_ids):
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            # token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_special_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.unk_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def token_ids_to_tensor(self, token_ids):
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id
        return torch.tensor(token_ids)

    def get_attn_mask(self, token_ids_tensor: T) -> T:
        attention_mask = torch.ones(token_ids_tensor.shape, dtype=torch.long, device=token_ids_tensor.device)
        attention_mask[token_ids_tensor == self.get_pad_id()] = 0
        return attention_mask

    def get_global_attn_mask(self, tokens_ids_tensor: T, gobal_mask_idxs) -> T:
        global_attention_mask = torch.zeros(tokens_ids_tensor.shape, dtype=torch.long, device=tokens_ids_tensor.device)
        global_attention_mask[gobal_mask_idxs] = 1
        return global_attention_mask

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

def get_hotpotqa_longformer_tokenizer(model_name=PRE_TAINED_LONFORMER_BASE, do_lower_case=True):
    tokenizer = LongformerTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    special_tokens_dict = {'additional_special_tokens': ['<q>', '</q>', '<d>', '<p>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('Number of added tokens = {}: {}'.format(num_added_toks, special_tokens_dict))
    return tokenizer

class LongformerEncoder(LongformerModel):
    def __init__(self, config, project_dim: int = 0, seq_project=True):
        LongformerModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.seq_project = seq_project
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, attn_dropout: float = 0.1,
                     hidden_dropout: float = 0.1, seq_project=False, **kwargs) -> LongformerModel:
        cfg = LongformerConfig.from_pretrained(cfg_name if cfg_name else PRE_TAINED_LONFORMER_BASE)
        if attn_dropout != 0:
            cfg.attention_probs_dropout_prob = attn_dropout
        if hidden_dropout !=0:
            cfg.hidden_dropout_prob = hidden_dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, seq_project=seq_project, **kwargs)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            attention_mask=attention_mask,
                                                                            global_attention_mask=global_attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             global_attention_mask=global_attention_mask)
        pooled_output = sequence_output[:, 0, :] ### get the first element [CLS], the second is the adding new token
        # print(pooled_output.shape, sequence_output.shape)
        if self.encode_proj:
            if self.seq_project:
                sequence_output = self.encode_proj(sequence_output)
                pooled_output = sequence_output[:, 0, :]
            else:
                pooled_output = self.encode_proj(pooled_output)
        # print(pooled_output.shape, sequence_output.shape)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

if __name__ == '__main__':
    tokenizer = LongformerTokenizer.from_pretrained(PRE_TAINED_LONFORMER_BASE, do_lower_case=True)

    x = '"spawn of the north"'
    y = 'sent  it is a remake of the 1938 film "spawn of the north".'

    print(tokenizer.encode(x, add_special_tokens=False))
    print(tokenizer.encode(y, add_special_tokens=False))
    # print(len(tokenizer))
    #
    # special_tokens_dict = {'additional_special_tokens': ['[C1]', '[C2]', '[C3]', '[C4]']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # tokenizer = get_hotpotqa_longformer_tokenizer()
    #
    # tokenierLongformer = LongformerQATensorizer(tokenizer=tokenizer, max_length=4096)
    #
    # x = '<q>hello world'
    # xids = tokenierLongformer.text_encode(x, add_special_tokens=False)
    # print(tokenizer.decode(xids))
    #
    # # tokenizer.add_special_tokens({'additional_special_tokens':})
    # print(tokenizer.special_tokens_map)
    # print(tokenizer.additional_special_tokens)
    # prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    # choice0 = "It is eaten with a fork and a knife."
    # choice1 = "It is eaten while held in the hand."
    # encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
    #
    # print(tokenizer.decode(encoding['input_ids'][1]))
    print()