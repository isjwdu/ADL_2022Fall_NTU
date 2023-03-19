import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
#coding=utf-8
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from tqdm import trange
import numpy as np
import random
import torch
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
import json
import os
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch import nn,optim
import pickle as pk
import shutil
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
else:
    device = torch.device('cpu')

parameter_copy = {

    'd_model':768,

    'hid_dim':300,

    'epoch':10,

    'batch_size':95,

    'n_layers':2,

    'dropout':0.1,

    'device':device,

    'lr':0.001,

    'momentum':0.99,

    'ckpt_path':'./ckpt/slot',
}


def build_dataSet(parameter):
    data_name = ['train', 'eval', 'test']

    data_set = {}
    key_table = defaultdict(int)
    vocab_table = defaultdict(int)

    vocab_table['<PAD>'] = 0
    vocab_table['<UNK>'] = 1
    key_table['O']

    for i in data_name:
        data_set[i] = []
        with open('./data/slot/' + i + '.json', 'r', encoding='utf-8') as f:
            data_src = json.load(f)
        for data in data_src:
            text = data['tokens']
            if i != 'test':
                label = data['tags']
                for key in label:
                    key_table[key] += 1

            for j in text:
                vocab_table[j] += 1

            if i != 'test':
                data_set[i].append([' '.join(text), label])
            else:
                data_set[i].append([' '.join(text)])

    key2ind = dict(zip(key_table.keys(), range(len(key_table))))
    ind2key = dict(zip(range(len(key_table)), key_table.keys()))

    word2ind = dict(zip(vocab_table.keys(), range(len(vocab_table))))
    ind2word = dict(zip(range(len(vocab_table)), vocab_table.keys()))
    parameter['key2ind'] = key2ind
    parameter['ind2key'] = ind2key
    parameter['word2ind'] = word2ind
    parameter['ind2word'] = ind2word
    parameter['data_set'] = data_set
    parameter['output_size'] = len(key2ind)
    parameter['word_size'] = len(word2ind)
    return parameter


def batch_yield_bert(parameter, shuffle=True, mode='train'):
    data_set = parameter['data_set'][mode]

    if shuffle:
        random.shuffle(data_set)

    if mode == 'train' or mode == 'eval':
        inputs, targets = [], []
        max_len = 0
        for items in tqdm(data_set):

            input = itemgetter(*(items[0].split()))(parameter['word2ind'])
            target = itemgetter(*items[1])(parameter['key2ind'])
            input = input if type(input) == type(()) else (input, 0)
            target = target if type(target) == type(()) else (target, 0)
            if len(input) > max_len:
                max_len = len(input)
            inputs.append(list(input))
            targets.append(list(target))
            if len(inputs) >= parameter['batch_size']:
                inputs = [i + [0] * (max_len - len(i)) for i in inputs]
                targets = [i + [0] * (max_len - len(i)) for i in targets]
                yield list2torch(inputs), list2torch(targets)
                inputs, targets = [], []
                max_len = 0

        inputs = [i + [0] * (max_len - len(i)) for i in inputs]
        targets = [i + [0] * (max_len - len(i)) for i in targets]
        yield list2torch(inputs), list2torch(targets)
        inputs, targets = [], []
        max_len = 0

    elif mode == 'test':
        inputs = []
        inputs_char = []
        for items in tqdm(data_set):

            input = itemgetter(*(items[0].split()))(parameter['word2ind'])
            if isinstance(input, int):
                inputs.append([input])
            else:
                inputs.append(list(input))
            yield list2torch(inputs)
            inputs = []

def list2torch(ins):
    try:
        return torch.from_numpy(np.array(ins))
    except:
        length = len(ins[0])
        for i in ins:

            if len(i) != length:
                i.pop(-1)
        return torch.from_numpy(np.array(ins))
'''
def predict_or_write(parameter=None):
    model = torch.load(args.ckpt_path)
    test_dataloader = batch_yield_bert(parameter, mode='test', shuffle=False)
    model.eval()

    result = pd.DataFrame(columns=['id', 'tags'])
    with open('./data/slot/test.json') as f:
        src_datas = json.load(f)
    for i, seqs in enumerate(test_dataloader):
        out = model(seqs.long().to(parameter['device']))
        _, predict = torch.max(out, dim=-1)
        predict_label = [parameter['ind2key'][predict_id.item()] for predict_id in predict]
        src_data = src_datas[i]
        result = result.append(dict(id=src_data['id'], tags=' '.join(predict_label)), ignore_index=True)

    result.to_csv(args.pred_file, index=False)
'''
def main(args):
    # TODO: implement main function
    model = torch.load(args.ckpt_path,map_location ='cpu')
    test_dataloader = batch_yield_bert(parameter, mode='test', shuffle=False)
    model.eval()

    result = pd.DataFrame(columns=['id', 'tags'])
    with open(args.test_file) as f:
        src_datas = json.load(f)
    for i, seqs in enumerate(test_dataloader):
        out = model(seqs.long().to(parameter['device']))
        _, predict = torch.max(out, dim=-1)
        predict_label = [parameter['ind2key'][predict_id.item()] for predict_id in predict]
        src_data = src_datas[i]
        result = result.append(dict(id=src_data['id'], tags=' '.join(predict_label)), ignore_index=True)

    result.to_csv(args.pred_file, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument(
        "--pred_file",
        type=Path,
        help="Predict file.",
        default="pred.slot.csv",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


class bilstm(nn.Module):
    def __init__(self, parameter):
        super(bilstm, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']

        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out)
        out = self.fc(out)
        return out.view(-1, out.size(-1))

if __name__ == "__main__":
    if not os.path.exists('parameter.pkl'):
        parameter = parameter_copy

        parameter = build_dataSet(parameter)
        pk.dump(parameter, open('parameter.pkl', 'wb'))
    else:

        parameter = pk.load(open('parameter.pkl', 'rb'))
        for i in parameter_copy.keys():
            if i not in parameter:
                parameter[i] = parameter_copy[i]
                continue
            if parameter_copy[i] != parameter[i]:
                parameter[i] = parameter_copy[i]
        for i in parameter_copy.keys():
            print(i, ':', parameter[i])
        pk.dump(parameter, open('parameter.pkl', 'wb'))
        del parameter_copy, i

    #predict_or_write(parameter=parameter)

    args = parse_args()
    main(args)