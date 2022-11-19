from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch
import os
from app_predict.model_custom import MyModel
import app_predict.utils as utils
import app_predict.config as config


class PredictDataset(Dataset):
    def __init__(self, sentences_list):
        super(PredictDataset, self).__init__()
        self.data = sentences_list
        self.tokenizer = utils.tokenizers

    def __getitem__(self, item):

        cur_data = self.data[item]
        inputs_a = self.tokenize_function(cur_data[0])
        inputs_b = self.tokenize_function(cur_data[1])
        return {'inputs_a': inputs_a, 'inputs_b': inputs_b}

    def __len__(self):
        return len(self.data)

    def tokenize_function(self, text):

        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=config.token_truncation,
                                max_length=config.token_max_length,
                                return_tensors=config.return_tensors)

        for _ in inputs:
            inputs[_] = inputs[_].to(config.device)
        return inputs


class SimilarityPredict():
    def __init__(self, model_path=config.best_model_path):

        self.model_path = model_path
        self.model = self.build_model()
        self.tokenizer = utils.tokenizers

    # sentences_sequence: [sen_a, sen_b, sen_c, sen_d, ...]
    def predict_sequence(self, sentences_sequence, window_size=config.similarity_window_size):

        # 构造为 sentences_list: [[sen_a, sen_b], [sen_a, sen_b]...]
        sentences_list = []
        # 指定 左右指针
        left = 0
        right = 0
        temp_list = []
        while right < len(sentences_sequence):
            if left == right:
                right = right + 1
                left = right - window_size + 1
                left = left if left >= 0 else 0
                if len(temp_list) != 0:
                    sentences_list.append(temp_list.copy())
                temp_list.clear()
                continue

            temp_list.append([sentences_sequence[left], sentences_sequence[right]])
            left = left + 1

        # 输入到 model 中
        similarity_list = []
        for one_sentences_list in sentences_list:
            similarity_list.append(self.predict_list(one_sentences_list))
        return similarity_list

    # len <= 7 [sen, sen, ...]
    def predict_one_sequence(self,sentences_sequence, similarity_wight=0, diff_wight=1):

        sentence_pair_list = []

        sequence_len = len(sentences_sequence)

        if sequence_len < 2:
            return

        for i in range(sequence_len - 1):
            sentence_pair_list.append([sentences_sequence[i], sentences_sequence[-1]])

        predict = self.predict_list(sentence_pair_list)

        # assert type(predict) == list

        node_similarity = 0
        for similarity_pair in predict:
            node_similarity = node_similarity + similarity_pair[0] * diff_wight + similarity_pair[1] * similarity_wight

        return predict, node_similarity

    # sentences_list: [[sen_a, sen_b], [sen_a, sen_b]...]
    def predict_list(self, sentences_list):

        results = []

        # 构建 dataset
        sentences_dataset = PredictDataset(sentences_list)

        # 构建 dataloader
        dataloader = DataLoader(dataset=sentences_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                drop_last=False)

        # 输入模型
        for index, batch in tqdm(enumerate(dataloader)):

            # 构建 dataset 的时候已经处理过了，这里只需要解引用即可
            # [batch_size, 2]
            similarity_out = self.model(**batch)
            similarity_out = torch.exp(similarity_out).cpu().detach().numpy()
            results.append(similarity_out)

        results = np.concatenate(results, axis=0)

        # predicts = np.argmax(results, axis=-1)

        return results.tolist()

    # 输入以两个句子构成 list 或 tuple 的形式输入
    def predict(self, sentence_pair):
        # 构建输入 dataset
        # inputs_dataset = PredictDataset(sentence_pair=sentence_pair)

        # 输入一个句子对 就不需要 构建 dataset 了
        # [1, seq_len]
        inputs_a = self.tokenize_function(sentence_pair[0])
        inputs_b = self.tokenize_function(sentence_pair[1])

        # 输入到模型
        # [1, 2]
        prediction = self.model(inputs_a=inputs_a, inputs_b=inputs_b)
        prediction = torch.exp(torch.squeeze(prediction)).cpu().detach().numpy().tolist()

        return prediction

    def build_model(self):
        model = MyModel()
        # print(self.model_path)
        model.load_state_dict(torch.load(self.model_path, map_location=config.device))
        model.to(config.device)
        model.eval()

        return model

    def tokenize_function(self, text):

        inputs = self.tokenizer(text,
                                padding='max_length',
                                truncation=config.token_truncation,
                                max_length=config.token_max_length,
                                return_tensors=config.return_tensors)

        for _ in inputs:
            inputs[_] = inputs[_].to(config.device)
        return inputs

    def computed_similarity(self, similarity_list, similarity_wight=0, diff_wight=1):
        similarity_group = []
        for node in similarity_list:
            node_similarity = 0
            for similarity_pair in node:
                node_similarity = node_similarity + similarity_pair[0] * diff_wight + similarity_pair[1] * similarity_wight
            similarity_group.append(node_similarity)
        return np.mean(similarity_group)
