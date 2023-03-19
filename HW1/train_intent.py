# coding=utf-8
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import shutil
import os
import torch
from tqdm import trange
from tqdm import tqdm

from dataset import SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader
from model import SeqClassifier
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from operator import itemgetter
import numpy as np
from sklearn.metrics import accuracy_score
import gensim
from collections import defaultdict
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import pickle as pk
import pandas as pd

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


# ----------------------------------------------------------------------
def batch2id(vocab, intent2id, batch, device):
    """"""
    max_len = 0
    chars = []
    labels = []
    for ind, text in enumerate(batch['text']):

        char_id = []
        if len(text) > max_len:
            max_len = len(text)
        text = text.split(' ')
        for char in text:
            try:
                char_id.append(vocab[char])
            except:
                char_id.append(vocab['[UNK]'])

        chars.append(char_id)
        labels.append(intent2id[batch['intent'][ind]])

    return torch.from_numpy(np.array([char + (max_len - len(char)) * [0] for char in chars])).to(
        device), torch.from_numpy(np.array(labels)).to(device).long()


def build_vocab(datas, args):
    chars = []

    vocab = defaultdict(int)
    vocab['<pad>'] = args.PAD
    vocab['<unk>'] = args.UNK

    for data in tqdm(datas):
        chars.append(data['text'].split())

        for char in chars[-1]:
            vocab[char] += 1
    # for char in vocab:
    # if vocab[char] < args.min_count_word:
    # del vocab[char]
    char2ind, ind2char = dict(zip(vocab.keys(), range(len(vocab)))), \
                         dict(zip(range(len(vocab)), vocab.keys()))
    args.char2ind = char2ind


def build_dataSet(datas, args, intent2idx, mode):
    if mode == 'train':
        # 構建數據集（chars，labels）
        chars = []
        labels = []

        for data in tqdm(datas):
            chars.append(data['text'].split())
            labels.append(intent2idx[data['intent']])

        char2ind = args.char2ind
        ind2embeding = np.random.randn(len(char2ind), args.embedding_dim).astype(np.float32) / np.sqrt(len(char2ind))
        # 加載詞向量
        glove_input_file = './glove.840B.300d.txt'
        word2vec_output_file = './glove.840B.300d.word2vec.txt'
        glove2word2vec(glove_input_file, word2vec_output_file)
        glove_model = KeyedVectors.load_word2vec_format('./glove.840B.300d.word2vec.txt', binary=False)
        # with open('glove_model.pkl', 'rb') as f:
        #    glove_model = pk.load(f)

        for ind, i in enumerate(char2ind.keys()):
            try:
                embedding = np.asarray(glove_model[i], dtype='float32')
                ind2embeding[ind] = embedding
            except:
                args.num_unknow += 1
                # print(i)

        args.ind2embeding = ind2embeding
        args.output_size = len(set(labels))

        return np.array(chars), np.array(labels)

    elif mode == 'dev':

        # 構建訓練集（chars，labels）
        chars = []
        labels = []

        for data in tqdm(datas):
            chars.append(data['text'].split())
            labels.append(intent2idx[data['intent']])

        return np.array(chars), np.array(labels)

def batch_yield(chars, labels, shuffle=True):
    if shuffle:
        permutation = np.random.permutation(len(chars))
        chars = chars[permutation]
        labels = labels[permutation]
    max_len = 0
    batch_x, batch_y, len_x = [], [], []
    for iters in tqdm(range(len(chars))):
        # batch_ids = itemgetter(*chars[iters])(args.char2ind)
        batch_ids = []
        for one_iter in chars[iters]:
            try:
                batch_ids.append(args.char2ind[one_iter])
            except:
                batch_ids.append(args.char2ind['<unk>'])
        try:
            batch_ids = list(batch_ids)
        except:
            batch_ids = [batch_ids, 0]
        if len(batch_ids) > max_len:
            max_len = len(batch_ids)
        batch_x.append(batch_ids)
        batch_y.append(labels[iters])

        if len(batch_x) >= args.batch_size:
            batch_xs = []
            batch_x = [np.array(
                list(itemgetter(*x_ids)(args.ind2embeding)) + [args.ind2embeding[0]] * (max_len - len(x_ids))) if len(
                x_ids) > 1 else np.array(
                [itemgetter(*x_ids)(args.ind2embeding)] + [args.ind2embeding[0]] * (max_len - len(x_ids))) for x_ids in
                       batch_x]

            yield torch.from_numpy(np.array(batch_x)).to(args.device), torch.from_numpy(np.array(batch_y)).to(
                args.device).long()
            max_len, batch_x, batch_y = 0, [], []
    batch_x = [
        np.array(list(itemgetter(*x_ids)(args.ind2embeding)) + [args.ind2embeding[0]] * (max_len - len(x_ids))) if len(
            x_ids) > 1 else np.array(
            [itemgetter(*x_ids)(args.ind2embeding)] + [args.ind2embeding[0]] * (max_len - len(x_ids))) for x_ids in
        batch_x]

    yield torch.from_numpy(np.array(batch_x)).to(args.device), torch.from_numpy(np.array(batch_y)).to(
        args.device).long()
    max_len, batch_x, batch_y = 0, [], []


def main(args):
    # record loss
    shutil.rmtree('textrnn+Attention') if os.path.exists('textrnn+Attention') else 1
    writer = SummaryWriter('./textrnn+Attention', comment='textrnn+Attention')

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # datasets: Dict[str, SeqClsDataset] = {
    # split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
    # for split, split_data in data.items()
    # }

    build_vocab(data['train'] + data['eval'], args=args)
    train_chars, train_label = build_dataSet(data['train'], args=args, intent2idx=intent2idx, mode='train')
    dev_chars, dev_labels = build_dataSet(data['eval'], args=args, intent2idx=intent2idx, mode='dev')
    # TODO: crecate DataLoader for train / dev datasets
    trian_dataloader = batch_yield(train_chars, train_label, shuffle=True)
    dev_dataloader = batch_yield(dev_chars, dev_labels, shuffle=False)

    # embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=args.output_size).to(args.device)
    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    model.train()
    train_step = 0
    dev_step = 0
    min_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for seqs, label in trian_dataloader:
            train_step += 1
            out = model(seqs)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            print('epoch [{}/{}], Train_Loss: {:.4f}'.format(train_step % len(list(trian_dataloader)), epoch,
                                                             loss.item()))
            writer.add_scalar('Train_loss', loss.item(), global_step=train_step)
        trian_dataloader = batch_yield(train_chars, train_label, shuffle=True)

        # TODO: Evaluation loop - calculate accuracy and save model weights
        # 進入評價模式
        model.eval()
        predict_label = []
        true_label = []
        for seqs, label in dev_dataloader:
            dev_step += 1
            out = model(seqs)
            _, predict = torch.max(out, dim=-1)
            loss = criterion(out, label)

            print('epoch [{}/{}], Dev_Loss: {:.4f}'.format(dev_step % len(list(dev_dataloader)), epoch, loss.item()))
            writer.add_scalar('Dev_loss', loss.item(), global_step=dev_step)

            predict_label += predict.flatten().tolist()
            true_label += label.flatten().tolist()

        acc = accuracy_score(true_label, predict_label)
        writer.add_scalar('Dev_acc', acc, global_step=epoch)
        print('epoch [{}/{}], Dev_Acc: {:.4f}'.format(epoch, args.num_epoch, acc))

        if acc > min_acc:
            min_acc = acc
            torch.save(model, args.ckpt_dir / 'my_rnn_attention.pt')
        dev_dataloader = batch_yield(dev_chars, dev_labels, shuffle=False)
    writer.flush()
    writer.close()

    # 寫入結果
    idx2intent = {v: k for k, v in intent2idx.items()}

    model = model.load('ckpt/intent/my_rnn_attention.pt')
    model.eval()
    with open('data/intent/test.json') as f:
        test_data = json.load(f)
    result = pd.DataFrame(columns=['id', 'intent'])

    for data in tqdm(test_data):
        text = data['text']
        id = data['id']
        batch_ids = []
        for one_iter in text.split():
            try:
                batch_ids.append(args.char2ind[one_iter])
            except:
                batch_ids.append(args.char2ind['<unk>'])
        if len(batch_ids) == 1:
            id2embedding = np.array([itemgetter(*batch_ids)(args.ind2embeding)])[None, :, :]
        else:
            id2embedding = np.array(list(itemgetter(*batch_ids)(args.ind2embeding)))[None, :, :]
        embedding2torch = torch.from_numpy(np.array(id2embedding)).to(args.device)
        _, idx = torch.max(model(embedding2torch), dim=-1)
        intent = idx2intent[idx.item()]
        result = result.append(dict(id=id, intent=intent), ignore_index=True)

    result.to_csv('data/intent/test.csv', index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--min_count_word", type=int, default=1)
    parser.add_argument("--num_unknow", type=int, default=0)

    parser.add_argument("--ind2char", type=dict)
    parser.add_argument("--char2ind", type=dict)
    parser.add_argument("--ind2embeding", type=float)
    parser.add_argument("--output_size", type=int, default=0)

    parser.add_argument("--PAD", type=int, default=1)
    parser.add_argument("--UNK", type=dict, default=2)

    # model
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=500)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)