from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from src.confs import envs_conf


@dataclass(frozen=True)
class DataProcessingSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    def text_file_reader(self, file_name: str) -> List[List[str]]:
        """Return a list of example that are: [word, V or N, sentence_example1, sentence_example2]"""
        texts = open(self.envs_conf.dev_data_dir / file_name, "r").read().splitlines()
        texts = [text.split("\t") for text in texts]
        print(len(texts[0]))
        print(type(texts[0]))
        cleaned_texts = []
        # Removing the position + Cleaning the extra space for each example.
        for word, verb_noun, _, example1, example2 in texts:
            cleaned_sample = [word, verb_noun, example1[:-2] + ".", example2[:-2] + "."]
            cleaned_texts.append(cleaned_sample)
        return cleaned_texts

    def word_level_bigram(self, dataset: List[List[str]]) -> torch.Tensor:
        raise NotImplementedError


impl = DataProcessingSvc()
