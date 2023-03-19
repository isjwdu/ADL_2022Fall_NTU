from typing import List
from itertools import chain
from dataclasses import dataclass, field

import torch
from transformers import (
    PreTrainedTokenizerBase,
)


@dataclass
class PreprocessCteSle:
    tokenizer: PreTrainedTokenizerBase
    return_label: bool = True
    max_len: int = 384
    padding: str = "max_length"
    question_name: str = "question"
    ending_names: List[str] = field(default_factory=
                                    lambda: [f"ending{i}" for i in range(4)])

    def __call__(self, examples):
        batch_size = len(examples[self.question_name])
        # Question
        first_sentences = [[context] * 4 for context in examples[self.question_name]]
        # Context
        second_sentences = [[examples[end][i] for end in self.ending_names] for i in range(batch_size)]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = self.tokenizer(
            second_sentences,
            first_sentences,
            max_length=self.max_len,
            padding=self.padding,
            truncation="only_first",
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        if self.return_label:
            labels = examples["label"]
            tokenized_inputs["labels"] = labels

        return tokenized_inputs


def data_collator(features):
    first = features[0]
    batch = {}

    ids = [feature.pop("id") for feature in features]
    paragraphs = [feature.pop("paragraphs") for feature in features]

    label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
    dtype = torch.long if isinstance(label, int) else torch.float
    batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["ids"] = ids
    batch["paragraphs"] = paragraphs

    return batch
