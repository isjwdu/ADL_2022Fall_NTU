import csv
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)
os.environ['TRANSFORMERS_CACHE'] = './cache'
from utils_qa import create_and_fill_np_array, PreprocessQAValid
from utils_ctx_sle import PreprocessCteSle, data_collator
from postprocess import postprocess_qa_predictions
accelerator = Accelerator()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ctx_sle_ckpt)

    ctx_sle_model = AutoModelForMultipleChoice.from_pretrained(args.ctx_sle_ckpt)
    ctx_sle_model.to(DEVICE)
    ctx_sle_model.resize_token_embeddings(len(tokenizer))

    raw_test_dataset = load_dataset("ctx_sle.py", name="test",
                                    question_file=args.test_file, context_file=args.ctx_file, cache_dir="./cache/ctx")

    preprocess_ctx_sle_fn = PreprocessCteSle(tokenizer, False, 512)
    test_dataset = raw_test_dataset["test"]
    with accelerator.main_process_first():
        test_dataset = test_dataset.map(
            preprocess_ctx_sle_fn,
            batched=True,
        )
    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=8)

    all_ids = []
    all_paragraphs = []
    all_pred = []

    for step, batch in enumerate(test_dataloader):
        ids = batch.pop("ids")
        paragraphs = batch.pop("paragraphs")
        _ = batch.pop("labels")
        data = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = ctx_sle_model(**data)
        predicted = outputs.logits.argmax(dim=-1)

        all_ids += ids
        all_paragraphs += paragraphs
        all_pred.append(predicted)

    all_pred = torch.cat(all_pred).cpu().numpy()
    selected_context = {instance_id: paragraphs[contex_idx]
                        for instance_id, contex_idx, paragraphs in zip(all_ids, all_pred, all_paragraphs)}
    print(selected_context, len(selected_context))
    tokenizer = AutoTokenizer.from_pretrained(args.qa_ckpt)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_ckpt)

    raw_test_dataset = load_dataset("qa.py", name="test",
                                    question_file=args.test_file, context_file=args.ctx_file, cache_dir="./cache/qa")

    column_names = raw_test_dataset["test"].column_names
    context_data = json.loads(args.ctx_file.read_text(encoding='utf-8'))

    def prepare_context(examples):
        """
        Fill the context according to the CtxSle model prediction
        """
        examples["context"] = [context_data[selected_context[idx]]
                               for idx in examples["id"]]
        return examples

    with accelerator.main_process_first():
        test_example = raw_test_dataset["test"].map(
            prepare_context,
            batched=True,
            desc="Running tokenizer on test dataset",
        )
    print("Test Example:", test_example[0])

    preprocess_qa_fn = PreprocessQAValid(tokenizer, 384, tokenizer.padding_side == "right")
    with accelerator.main_process_first():
        test_dataset = test_example.map(
            preprocess_qa_fn,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on test dataset",
        )
    print("Test Dataset:", test_dataset[0])

    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    qa_data_collator = default_data_collator
    test_dataloader = DataLoader(
        test_dataset_for_model, shuffle=False, collate_fn=qa_data_collator, batch_size=8
    )
    model, test_dataloader = accelerator.prepare(
        qa_model, test_dataloader
    )

    all_start_logits = []
    all_end_logits = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = qa_model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)

    predictions = postprocess_qa_predictions(
        examples=test_example,
        features=test_dataset,
        predictions=outputs_numpy,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=None,
        prefix="eval",
    )

    with open(args.pred_file, 'w', newline='', encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'answer'])

        for example_id, answer in predictions.items():
            writer.writerow((example_id, answer))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=Path, default="./data/test.json")
    parser.add_argument("--ctx_file", type=Path, default="./data/context.json")
    parser.add_argument("--pred_file", type=Path, default="result.csv")
    parser.add_argument("--ctx_sle_ckpt", type=Path, default="./ckpt/ctx_sle")
    parser.add_argument("--qa_ckpt", type=Path, default="./ckpt/qa")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arges = parse_args()
    main(arges)
