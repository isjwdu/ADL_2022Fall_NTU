# README
Since the compressed file is too large for the operation, it does not contain the 840B.300D.txt file, so you need to put this file in the root directory before running train_*.py. In addition, the 840B.300D.zip file cannot be decompressed from the command line on my computer, so you need to decompress it manually.
### Environment
```
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.in
```
### Preprocessing
```
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent
### Train and Predict
```
python train_intent.py
```
I write the training program and prediction program in same .py file (train_intent.py), just run train_intent to train the .pt and get .csv file.

### Predict(Single)
```
python test_intent.py --test_file "${1}" --ckpt_path "${2}"
```
- ckpt_path:model path
- test_file:test_json path

### Reproduce
```
bash download.sh
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot
```
python train_slot.py
```
I write the training program and prediction program in same file (train_intent.py), just run train_intent to train the .pt and get .csv file.

### Predict(Single)
```
python test_slot.py --test_file "${1}" --ckpt_path "${2}"
```
- ckpt_path:model path
- test_file:test_json path

### Reproduce
```
bash download.sh
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```
