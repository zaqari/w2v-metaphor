import torch
import torch.nn as nn
from transformers import *

class bert(nn.Module):

    def __init__(self, model_type='bert-base-uncased'):
        super(bert, self).__init__()
        self.bert = BertModel.from_pretrained(model_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

    def translate(self, input_text, token_type_ids=None, attention_mask=None, labels=None):
        self.bert.eval()

        tokenized_text = self.tokenizer.tokenize(input_text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        embedded_data, pooled_output = self.bert(torch.tensor(input_ids).view(-1, 1), token_type_ids, attention_mask)

        return embedded_data, pooled_output