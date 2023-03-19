# README

## Before Training
You need to install packages/tools/dependencies ... below:
```
Python 3.8 / 3.9 and Python Standard Library
PyTorch 1.12.1, TensorFlow 2.10.0
Tqdm,numpy, pandas, scikit-learn 1.1.2, nltk 3.7
transformers==4.22.2, datasets==2.5.2, accelerate==0.13.0
```
After that, If you want to train by yourself, you have to download Chinese Pretrain-Model from https://huggingface.co/, like macbert-chinese/roberta-chinese/based-bert-chinese etc.

## Train Multiple_Choice Task
```
python train_ctx_sle.py --ckpt_dir --max_len --model_name_or_path --lr --weight_decay --batch_size --gradient_accumulation_steps --num_epoch
```
--ckpt_dir: the dir that you want to safe your final model
--model_name_or_path: the path your pretrain model
--num_epoch: how many epoch you want to train
--max_len: suggest 512
the others we suggest to use defualt number 

## Train Question_Answer Task
```
python train_qa.py --data_dir --ckpt_dir --model_name_or_path --batch_size --num_epoch --lr_scheduler_type --with_tracking
```
--ckpt_dir: the dir that you want to safe your final model
--model_name_or_path: the path your pretrain model
--num_epoch: how many epoch you want to train
--max_len: suggest 512
the others we suggest to use defualt configuration

## Predict Question_Answer Task
```
python test_qa.py --test_file --ctx_file --pred_file --ctx_sle_ckpt --qa_ckpt
```
--test_file: the test.json in dataset path
--ctx_file: the context.json in dataset path
--pred_file: the path you want to save your predict.csv file 
--ctx_sle_ckpt: load the Multiple Choice task ckpt path
--qa_ckpt: the final QA Model path after you done Train QA Task

## Reproduce results
IMPORTANT: Because all project I have done is based on TWCC, so it could be very difficult for me to cut down the network to check run.sh. Because I knew the main download in test_qa.py is .cache folder, I save the cache folder and change the path in a zip file. Until 2022.11.10, I can make sure run.sh don't need to download anything in TWCC.ai (because I saved the cache file). But I still cannot make sure it doesn't need internet when it's running on TAs machines.
![](https://i.imgur.com/b4UD5Ka.png)

```
#Public Score is 0.79566 on Kaggle
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```
