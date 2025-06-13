# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List

# import torch

# from src.confs import envs_conf
# from src.modules import data_processing_svc


# class CharacterLevelBigramSvc:
#     envs_conf: envs_conf.EnvsConf = envs_conf.impl
#     data_processing_svc: data_processing_svc.DataProcessingSvc = (
#         data_processing_svc.impl
#     )

#     def __init__(self, file_name: str):
#         self.dataset = self.data_processing_svc.text_file_reader(file_name)
#         self.stoi = self.data_processing_svc.get_vocab_character(self.dataset)
#         self.itos = {i: s for s, i in self.stoi.items()}

#         self.N = self.character_level_bigram(self.dataset, self.stoi)
#         self.P = self.N.float()
#         self.P /= self.P.sum(1, keepdim=True)

#     def character_level_bigram(
#         self, dataset: List[List[str]], stoi: Dict[str, int]
#     ) -> torch.Tensor:
#         N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)
#         for data in dataset[:1]:
#             # word, V or N, example1, example2
#             _, _, example1, example2 = data

#             for word1, word2 in zip(word_list2, word_list2[1:]):
#                 ix1, ix2 = stoi[word1], stoi[word2]
#                 N[ix1, ix2] += 1
#         return N

#     def predict_next_character(self, word: str) -> str:
#         ix = self.stoi[word]
#         g = torch.Generator().manual_seed(42)
#         output = []
#         while True:
#             p = self.P[ix]
#             ix = torch.multinomial(p, num_samples=100, replacement=True, generator=g)
#             print(ix)
#             for i in ix:
#                 print(self.itos[int(i.item())])
#             break
#         return " ".join(output)
