from typing import Dict
import torch
from torch.nn import Embedding
import torch.nn as nn
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
            num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # embedding  from outside
        # self.embed = Embedding.from_pretrained(embeddings, freeze=False)

        # TODO: model architecture

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)
        self.fc_attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_class)
        self.softmax = nn.Softmax(dim=-1)

        # attention
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        # x = self.embed(batch)
        out, (h, c) = self.lstm(batch)
        #   print(self.fc_attention(out).shape,self.w.shape,'test')
        alpha = F.softmax(torch.matmul(F.tanh(self.fc_attention(out)), self.w), dim=1).unsqueeze(-1)
        # print(alpha.shape)
        # alpha = F.softmax(torch.matmul(F.tanh(out),self.w),dim = 1).unsqueeze(-1)
        # print(out.shape,alpha.shape)
        # out = F.relu(torch.sum(out * alpha,1))
        out = F.relu((out * alpha)[:, 0])
        out = self.fc(out)
        out = self.softmax(out)
        #  print(out.shape)
        return out


class SeqTagger(SeqClassifier):

    def __init__(self, args):
        super(bilstm, self).__init__()
        word_size = args['word_size']
        embedding_dim = args['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = args['hid_dim']
        num_layers = args['n_layers']
        dropout = args['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout)

        output_size = args['output_size']
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out = self.embedding(x)
        out, (h, c) = self.lstm(out)
        out = self.fc(out)
        return out.view(-1, out.size(-1))
