from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from src.confs import envs_conf
from src.modules import data_processing_svc


class BasicNeuralNetSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    data_processing_svc: data_processing_svc.DataProcessingSvc = (
        data_processing_svc.impl
    )

    def __init__(self, file_name: str):
        self.dataset = self.data_processing_svc.text_file_reader(file_name)
        self.stoi = self.data_processing_svc.get_vocab_word(self.dataset)
        self.itos = {i: s for s, i in self.stoi.items()}
        self.training_set = self.create_training_set(self.dataset, self.stoi)
        self.xenc = self.one_hot_encoding(self.training_set[0], self.stoi)

    def create_training_set(self, dataset: List[List[str]], stoi: Dict[str, int]):
        xs, ys = [], []
        for data in dataset[:1]:
            _, _, example1, example2 = data
            word_list1 = example1.split()
            word_list2 = example2.split()
            for word1, word2 in zip(word_list1, word_list1[1:]):
                ix1, ix2 = stoi[word1], stoi[word2]
                xs.append(ix1)
                ys.append(ix2)

            for word1, word2 in zip(word_list2, word_list2[1:]):
                ix1, ix2 = stoi[word1], stoi[word2]
                print(word1, word2)
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def one_hot_encoding(self, xs: torch.Tensor, stoi: Dict[str, int]):
        return F.one_hot(xs, num_classes=len(stoi)).float()
