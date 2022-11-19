import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from app_predict.model_custom import MyModel
import app_predict.config as config


class CosSimilarityPredict:

    def __init__(self, check_point="voidful/albert_chinese_tiny"):
        self.model = AutoModel.from_pretrained(check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)

    def predict_one_sequence(self,sentences_sequence):

        sentence_pair_list = []

        sequence_len = len(sentences_sequence)

        if sequence_len < 2:
            return

        for i in range(sequence_len - 1):
            sentence_pair_list.append([sentences_sequence[i], sentences_sequence[-1]])

        cos_similarity_list = self.predict_list(sentence_pair_list)
        cos_similarity = np.array(cos_similarity_list).sum()
        cos_similarity = float(cos_similarity)

        return cos_similarity_list, cos_similarity

    # sentences_list: [[sen_a, sen_b], [sen_a, sen_b]...]
    def predict_list(self, sentences_list):

        res = []
        for sentences_pair in sentences_list:
            res.append(self.predict(sentences_pair))
        return res

    # 输入以两个句子构成 list 或 tuple 的形式输入
    def predict(self, sentence_pair):

        cls_a = self.get_cls(sentence_pair[0])
        cls_b = self.get_cls(sentence_pair[1])

        # res = float(F.cosine_similarity(cls_a, cls_b, dim=0).detach().numpy())
        return (1 - F.cosine_similarity(cls_a, cls_b, dim=0).detach().numpy()) / 2
        # return  F.cosine_similarity(cls_a, cls_b, dim=0).detach().numpy()

    def get_cls(self, sen):
        inputs = self.tokenize_function(sen)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(1)[0]

    def tokenize_function(self, text):

        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=config.token_truncation,
                                max_length=config.token_max_length,
                                return_tensors=config.return_tensors)

        for _ in inputs:
            inputs[_] = inputs[_].to(config.device)
        return inputs
