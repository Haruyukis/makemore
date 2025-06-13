from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.confs import envs_conf


@dataclass(frozen=True)
class DataProcessingSvc:
    envs_conf: envs_conf.EnvsConf = envs_conf.impl

    def text_file_reader(self, file_name: str) -> List[List[str]]:
        """Return a list of example that are: [word, V or N, sentence_example1, sentence_example2]"""
        texts = open(self.envs_conf.dev_data_dir / file_name, "r").read().splitlines()
        texts = [text.split("\t") for text in texts]
        cleaned_texts = []
        for word, verb_noun, _, example1, example2 in texts:
            cleaned_sample = [
                word,
                verb_noun,
                example1,
                example2,
            ]
            cleaned_texts.append(cleaned_sample)
        return cleaned_texts

    def get_vocab_word(self, dataset: List[List[str]]) -> Dict[str, int]:
        example1 = set(" ".join([data[2] for data in dataset]).split())
        example2 = set(" ".join([data[3] for data in dataset]).split())

        # concatenated_example2 = [data[3] for data in dataset]
        # words = set("".join(concatenated_example1).split())
        example1.update(example2)
        words = sorted(list(example1))
        stoi = {s: i + 2 for i, s in enumerate(words)}
        stoi["<S>"] = 0
        stoi["<E>"] = 1

        return stoi

    def get_vocab_character(self, dataset: List[List[str]]) -> Dict[str, int]:
        example1 = set(" ".join([data[2] for data in dataset]).split())
        example2 = set(" ".join([data[3] for data in dataset]).split())
        example1.update(example2)
        full_text = " ".join(example1)
        characters = sorted(list(set(full_text)))
        stoi = {s: i + 1 for i, s in enumerate(characters)}
        return stoi


impl = DataProcessingSvc()
