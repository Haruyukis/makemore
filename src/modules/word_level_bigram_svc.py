from __future__ import annotations

from typing import Dict, List

import torch

from src.confs import envs_conf
from src.modules import data_processing_svc


class WordLevelBigramSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl
    data_processing_svc: data_processing_svc.DataProcessingSvc = (
        data_processing_svc.impl
    )

    def __init__(self, file_name: str):
        self.dataset = self.data_processing_svc.text_file_reader(file_name)
        self.stoi = self.data_processing_svc.get_vocab_word(self.dataset)
        self.itos = {i: s for s, i in self.stoi.items()}

        self.N = self.word_level_bigram(self.dataset, self.stoi)
        self.P = (self.N).float()
        self.P /= self.P.sum(1, keepdim=True)

    def word_level_bigram(
        self, dataset: List[List[str]], stoi: Dict[str, int]
    ) -> torch.Tensor:
        N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
        for data in dataset:
            # word, V or N, example1, example2
            _, _, example1, example2 = data
            example1 = "<S> " + example1 + " <E>"
            example2 = "<S> " + example2 + " <E>"
            word_list1 = example1.split()
            word_list2 = example2.split()
            for word1, word2 in zip(word_list1, word_list1[1:]):
                ix1, ix2 = stoi[word1], stoi[word2]
                N[ix1, ix2] += 1

            for word1, word2 in zip(word_list2, word_list2[1:]):
                ix1, ix2 = stoi[word1], stoi[word2]
                N[ix1, ix2] += 1
        return N

    def predict_next_word(self, word: str) -> str:
        ix = self.stoi[word]
        g = torch.Generator().manual_seed(42)
        output = []
        while True:
            p = self.P[ix]
            ix = torch.multinomial(p, num_samples=100, replacement=True, generator=g)
            print(ix)
            for i in ix:
                print(self.itos[int(i.item())])
            break
        return " ".join(output)

    def compute_log_likelihood(self):
        log_likelihood = 0.0
        n = 0
        for data in self.dataset:
            _, _, example1, example2 = data
            example1 = "<S> " + example1 + " <E>"
            example2 = "<S> " + example2 + " <E>"
            word_list1 = example1.split()
            word_list2 = example2.split()
            for word1, word2 in zip(word_list1, word_list1[1:]):
                ix1, ix2 = self.stoi[word1], self.stoi[word2]
                prob = self.P[ix1, ix2]
                logprobs = torch.log(prob)
                log_likelihood += logprobs
                n += 1

            for word1, word2 in zip(word_list2, word_list2[1:]):
                ix1, ix2 = self.stoi[word1], self.stoi[word2]
                prob = self.P[ix1, ix2]
                logprobs = torch.log(prob)
                log_likelihood += logprobs
                n += 1
        print(f"{log_likelihood=}")
        nll = -log_likelihood
        print(f"{nll=}")
        return nll / n
