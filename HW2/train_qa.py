import math
import uuid
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from postprocess import postprocess_qa_predictions
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed
)
from utils_qa import (
    PreprocessQATrain,
    PreprocessQAValid,
    create_and_fill_np_array,
    post_processing_function,
    save_log
)

UID = str(uuid.uuid1())


def main(args):
    set_seed(1)
    accelerator = Accelerator()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    raw_train_dataset = load_dataset("qa.py", name="train", cache_dir="./cache2",
                                     question_file="./data/train.json", context_file="./data/context.json")
    raw_valid_dataset = load_dataset("qa.py", name="eval", cache_dir="./cache2",
                                     question_file="./data/valid.json", context_file="./data/context.json")

    column_names = raw_train_dataset["train"].column_names

    preprocess_train_fn = PreprocessQATrain(tokenizer, args.max_len, tokenizer.padding_side == "right")
    preprocess_valid_fn = PreprocessQAValid(tokenizer, args.max_len, tokenizer.padding_side == "right")

    train_dataset = raw_train_dataset["train"]
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            preprocess_train_fn,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

    eval_example = raw_valid_dataset["validation"]
    with accelerator.main_process_first():
        eval_dataset = eval_example.map(
            preprocess_valid_fn,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

    print("Training set sample:", train_dataset[0])
    print("Eval set sample:", eval_dataset[0])
    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, shuffle=False, collate_fn=data_collator, batch_size=8
    )

    metric = load_metric("squad")
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epoch * num_update_steps_per_epoch
    print("Num iteration:", len(train_dataloader))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, ncols=100)
    completed_steps = 0
    log = {
        "train_loss": [],
        "valid_em": [],
        "valid_f1": [],
    }

    for epoch in range(args.num_epoch):
        model.train()
        running_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            running_loss += loss.item()
            if step % 1000 == 0 or step == len(train_dataloader) - 1:
                model.eval()
                all_start_logits = []
                all_end_logits = []
                for valid_step, valid_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**valid_batch)
                        start_logits = outputs.start_logits
                        end_logits = outputs.end_logits
                        start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                        end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                        all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                        all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

                max_len = max([x.shape[1] for x in all_start_logits])  
                start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
                end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

                del all_start_logits
                del all_end_logits

                outputs_numpy = (start_logits_concat, end_logits_concat)
                prediction = post_processing_function(eval_example, eval_dataset, outputs_numpy,
                                                      output_dir=args.ckpt_dir)
                eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

                log["train_loss"].append(running_loss / (step + 1))
                log["valid_em"].append(eval_metric["exact_match"])
                log["valid_f1"].append(eval_metric["f1"])

                model.train()

            if completed_steps >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.ckpt_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.ckpt_dir)

    save_log(log, os.path.join(args.ckpt_dir, "log.json"))
    print(f"Eval metric: em: {log['valid_em'][-1]}, f1: {log['valid_f1'][-1]}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/qa/",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=6,
        help="Num worker for preprocessing"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="",
        default="",
    )
    # data
    parser.add_argument("--max_len", type=int, default=384)

    # model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="hfl/chinese-roberta-wwm-ext",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # training
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--with_tracking", type=bool, default=True)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir = Path(f"{args.ckpt_dir}/{UID[:8]}")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
