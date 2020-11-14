import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import logging
import os
from transformers import LongformerForQuestionAnswering, LongformerTokenizer
import torch
from multihopUtils.longformerQAUtils import get_hotpotqa_longformer_tokenizer

TRIVIAQA_MODEL_NAME_LARGE = 'allenai/longformer-large-4096-finetuned-triviaqa'
SQUADV_MODEL_NAME = 'valhalla/longformer-base-4096-finetuned-squadv1'

def load_question_answer_model(qa_model_name: str=SQUADV_MODEL_NAME, return_dict=True):
    model = LongformerForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=qa_model_name, return_dict=return_dict)
    return model

if __name__ == '__main__':
    model_name = SQUADV_MODEL_NAME
    tokenizer = get_hotpotqa_longformer_tokenizer(model_name=model_name)
    model = load_question_answer_model(qa_model_name=model_name)
    model.resize_token_embeddings(len(tokenizer))
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    encoding = tokenizer(question, text, return_tensors="pt")
    print(encoding)
    input_ids = encoding["input_ids"]
    # # default is local attention everywhere
    # # the forward method will automatically set global attention on question tokens
    attention_mask = encoding["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask)
    print(outputs)
    # print(type(outputs))
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    print(start_logits.shape, end_logits.shape, input_ids.shape, '*'*10)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_logits):torch.argmax(end_logits) + 1]
    print(answer_tokens)
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    print(answer)


    #++++++++++++++++++++++++++++++++++++++++++++