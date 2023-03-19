# README
## For TA to execute
```
# Download the eval-summarization folder
bash ./download.sh

# Get the predicted file
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

# Eval the ROUGE Score
python3 eval.py -r path/to/test.jsonl -s ./submission.jsonl
```

## Before execute this code you need to know

You need to go to hugging face to download the mt5-small model and save it in the target folder. You can get more infomation from [this link](https://huggingface.co/google/mt5-small/tree/main)

## How to Train
```
python ./run_summarization.py --train_file path/to/train.jsonl --validation_file path/to/val.jsonl --model_name_or_path ./mt5-small --output_dir ./eval-summarization  --text_column maintext --summary_column title --do_train --num_train_epochs 5
```
--model_name_or_path: the path of saved mt5-small model   
--output_dir: the path of trained model you want to save in.   
## How to predict the jsonl file
```
python ./run_summarization.py --model_name_or_path ./eval-summarization --output_dir ./inference --do_predict --test_file path/to/test.jsonl --output_file ./submission.jsonl --text_column maintext --summary_column title --predict_with_generate --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_beams 10
```
--model_name_or_path: the path of trained model you have saved in.  
--output_dir: the path to save the config of your predict results.   
--num_beams: how many word you want beam search to execute.   
If you want to use top k or top p or temperature... you HAVE TO add --do_sample   
Eg.
```
python ./run_summarization.py --model_name_or_path ./eval-summarization --output_dir ./inference --do_predict --test_file ./data/public.jsonl --output_file ./test.jsonl --text_column maintext --summary_column title --predict_with_generate --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --temperature 0.5 --do_sample
```

## ROUGE score Install

`pip install` in ROUGE repo

## Usage

```
>>> from tw_rouge import get_rouge
>>> get_rouge('我是人', '我是一個人')
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], [ 我是一個人'])
{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}
>>> get_rouge(['我是人'], ['我是一個人'], avg=False)
[{'rouge-1': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}, 'rouge-2': {'f': 0.33333332888888895, 'p': 0.5, 'r': 0.25}, 'rouge-l': {'f': 0.7499999953125, 'p': 1.0, 'r': 0.6}}]
```

## How to eval rouge score
```
python3 eval.py -r path/to/test.jsonl -s path/to/your/submission.jsonl
```
