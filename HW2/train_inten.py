import math
import json
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    AutoModelForSequenceClassification,
    default_data_collator,
    AdamW,
    get_scheduler,
)

UID = str(uuid.uuid1())
accelerator = Accelerator()

def main(args):
    # 1. Prepare model
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=150)
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=150, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    intent_idx_path = Path("../HW1/cache/intent/intent2idx.json")
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    model.config.label2id = {l: i for i, l in enumerate(intent2idx)}
    model.config.id2label = {idx: label for label, idx in config.label2id.items()}

    # 2. Process data
    raw_datasets = load_dataset("seq_cls.py", name="train", cache_dir="./cache",
                                train_file='../HW1/data/intent/train.json',
                                eval_file='../HW1/data/intent/eval.json')

    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples["text"],)
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_len, truncation=True)

        if "intent" in examples:
            result["labels"] = [intent2idx[label] for label in examples["intent"]]
        # if "id" in examples:
        #     result["ids"] = [idx for idx in examples["id"]]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    print("Train data sample", train_dataset[0])

    data_collator = default_data_collator
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)

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

    # 3. Scheduler and math around the number of training steps.
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
    metric = load_metric("accuracy")
    # Training loop
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

            if completed_steps >= max_train_steps:
                break
        # Evaluation
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader):
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}: {eval_metric}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.ckpt_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.ckpt_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/seq_cls/",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=6,
        help="Num worker for preprocessing"
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="bert-base-cased",
    )

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    # data loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # training
    parser.add_argument("--num_epoch", type=int, default=3)
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
