# -*- coding: utf-8 -*-
"""
# Download dataset, build vocabulary and preprocess data
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
from collections import defaultdict

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class Voc:
    def __init__(self):
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.wcounts = defaultdict(lambda: 0)
        self.PAD = self.w2i["<pad>"]
        self.UNK = self.w2i["<unk>"]
        self.char2i = defaultdict(lambda: len(self.char2i))
        self.PADCHAR = self.char2i["<pad>"]
        self.UNKCHAR = self.char2i["<unk>"]
        #         [self.char2i[c] for c in 'abcdefghijklmnopqrstuvwxyz,.?-!'] # assumes all lowercase
        #         self.char2i = defaultdict(lambda: self.UNKCHAR, self.char2i)
        self.char_vocab_len = len(self.char2i)

    def add_sentence(self, line):
        # returns list of indices
        line = line.strip()
        words = []
        for w in line.split():
            words.append(self.w2i[w])
            [self.char2i[c] for c in w]
            self.wcounts[w] += 1
        return words

    def return_indices(self, line):
        line = line.strip()
        words = []
        words_char_level = []
        for w in line.split():
            if w in self.w2i:
                words.append(self.w2i[w])
            else:
                words.append(self.UNK)
            chars = []
            for c in w:
                if c in self.char2i:
                    chars.append(self.char2i[c])
                else:
                    chars.append(self.UNKCHAR)
            words_char_level.append(np.array(chars))
        return np.array(words), np.array(words_char_level)

    def trim(self, min_count):
        to_keep = []
        for w in self.wcounts:
            if self.wcounts[w] >= min_count:
                to_keep.append(w)
        self.w2i = {}
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.wcounts_updated = defaultdict(lambda: 0)
        self.PAD = self.w2i["<pad>"]
        self.UNK = self.w2i["<unk>"]
        for w in to_keep:
            self.w2i[w]
            self.wcounts_updated[w] = self.wcounts[w]
        self.wcounts = self.wcounts_updated

    def __len__(self):
        return len(self.w2i)


train_file = 'topicclass/topicclass_train.txt'
dev_file = 'topicclass/topicclass_valid.txt'
test_file = 'topicclass/topicclass_test.txt'

voc = Voc()


def build_vocab(file):
    # initial processing to be done on training set for building the vocab
    with open(file) as f:
        for line in f:
            _, line = line.lower().split('|||')
            voc.add_sentence(line)


def load_data(file):
    # this processes sentences and returns data in required format
    out = []
    topics = set()
    with open(file) as f:
        for line in f:
            topic, line = line.split('|||')
            topic = topic.strip()
            words, words_char_level = voc.return_indices(line.lower())
            out.append((topic, words_char_level, words))
            topics.add(topic)
    return out, topics


build_vocab(train_file)
print(len(voc))

voc.trim(5)
len(voc)
nwords = len(voc)

train_data, train_topics = load_data(train_file)
dev_data, dev_topics = load_data(dev_file)
all_topics = train_topics.union(dev_topics)

topic_to_idx = {topic: idx for (idx, topic) in enumerate(all_topics)}
idx_to_topic = {idx: topic for (topic, idx) in topic_to_idx.items()}
ntags = len(topic_to_idx)
idx_to_topic


class SentencesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns a tuple (words, words_char_level, labels
        sample = (torch.from_numpy(self.data[idx][2]),
                  torch.nn.utils.rnn.pad_sequence([torch.from_numpy(x) for x in self.data[idx][1]], batch_first=True)
                  , topic_to_idx[self.data[idx][0]])
        return sample


def pad_vect(vect, pad_len, dim):
    # vec padded to pad in dimension dim
    pad_amount = list(vect.shape)
    pad_amount[dim] = pad_len - vect.size(dim)
    padded = torch.cat([vect, torch.zeros(*pad_amount).long()], dim=dim)
    return padded


def collate_fn(batch):
    # takes a list of (tensor, label), returns padded examples and labels
    # find longest sentence
    dim = 0
    max_len = max(map(lambda x: x[0].shape[dim], batch))
    max_word_len = max(map(lambda x: x[1].shape[1], batch))
    batch = list(map(lambda d: (pad_vect(d[0], pad_len=max_len, dim=dim),
                                pad_vect(pad_vect(d[1], pad_len=max_word_len, dim=1), pad_len=max_len, dim=dim), d[2]),
                     batch))
    ws = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
    cs = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
    ys = torch.LongTensor(list(map(lambda x: x[2], batch)))
    return ws, cs, ys


train_samples = SentencesDataset(train_data)
dataloader_args = dict(shuffle=True, batch_size=512, num_workers=10, pin_memory=True, collate_fn=collate_fn) if cuda \
    else dict(shuffle=False, batch_size=64, collate_fn=collate_fn)
train_loader = dataloader.DataLoader(train_samples, **dataloader_args)


def train_epoch(model, train_loader, criterion, optimizer):
    # runs the training
    model.train()
    model.to(device)

    run_loss = 0.0
    correct_predictions = 0.0
    total_predictions = 0.0

    start_time = time.time()
    for batch_idx, sample in enumerate(train_loader):
        if batch_idx % 300 == 0:
            print(".", end='')
        optimizer.zero_grad()
        data_words = sample[0].to(device)
        data_chars = sample[1].to(device)
        target = sample[2].to(device)

        output = model(data_words, data_chars)
        _, predictions = torch.max(output.data, 1)
        correct_predictions += (predictions == target).sum().item()
        total_predictions += target.size(0)

        loss = criterion(output, target)
        run_loss += loss.item()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    acc = (correct_predictions / total_predictions) * 100.0
    run_loss /= len(train_loader)
    print('Train Loss: ', run_loss, '. Train Accuracy: ', acc, '%', 'Time: ', end_time - start_time, 's')
    return run_loss, acc


def test_model(model, test_loader, criterion):
    # runs validation, calculates accuracy and loss
    with torch.no_grad():
        model.eval()
        model.to(device)

        run_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0

        for _, sample in enumerate(test_loader):
            data_words = sample[0].to(device)
            data_chars = sample[1].to(device)
            target = sample[2].to(device)

            output = model(data_words, data_chars)
            _, predictions = torch.max(output.data, 1)
            correct_predictions += (predictions == target).sum().item()
            total_predictions += target.size(0)

            loss = criterion(output, target).detach()
            run_loss += loss.item()

        acc = (correct_predictions / total_predictions) * 100.0
        run_loss /= len(test_loader)
        print('Test Loss: ', run_loss, '. Test Accuracy: ', acc, '%')
        return run_loss, acc


def output_results(model, test_loader, criterion):
    # the output predictions for test
    output = []
    with torch.no_grad():
        model.eval()
        model.to(device)
        for _, sample in enumerate(test_loader):
            data_words = sample[0].to(device)
            data_chars = sample[1].to(device)
            outputs = model(data_words, data_chars)
            _, predicted = torch.max(outputs.data, 1)
            output.extend(predicted.cpu().numpy())
        return output


"""# Pretrained word embeddings"""

# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip

from emb_weights import EmbWeights

ew = EmbWeights(r"glove.6B.200d.txt")
emb_mat = ew.create_emb_matrix(voc.w2i)
emb_mat = torch.tensor(emb_mat).float()

"""# Create model and train"""

torch.backends.cudnn.deterministic = True
torch.manual_seed(5)

EMB_SIZE = 200
CHAR_EMB_SIZE = 20
CHAR_CNN_FILTER_SIZE = 32
RNN_EMB_SIZE = 200
WIN_SIZE = 3
FILTER_SIZE_1 = 64
FILTER_SIZE_2 = 128
FILTER_SIZE_3 = 256
FILTER_SIZE_4 = 256
DENSE_SIZE = 128


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.char_emb = nn.Embedding(len(voc.char2i), CHAR_EMB_SIZE, padding_idx=voc.PADCHAR)
        self.emb = nn.Embedding.from_pretrained(emb_mat, freeze=True, sparse=False)

        self.cnn_char = torch.nn.Conv1d(in_channels=CHAR_EMB_SIZE, out_channels=CHAR_CNN_FILTER_SIZE, kernel_size=3,
                                        stride=1, padding=3 // 2, dilation=1, groups=1, bias=True)

        self.gru_emb = torch.nn.GRU(input_size=EMB_SIZE + CHAR_CNN_FILTER_SIZE, hidden_size=RNN_EMB_SIZE // 2,
                                    num_layers=2, bidirectional=True)

        self.conv3 = torch.nn.Conv1d(in_channels=RNN_EMB_SIZE, out_channels=FILTER_SIZE_3, kernel_size=WIN_SIZE,
                                     stride=2, padding=WIN_SIZE // 2, dilation=1, groups=1, bias=True)
        self.mpool3 = nn.MaxPool1d(WIN_SIZE, padding=WIN_SIZE // 2)

        self.conv4 = torch.nn.Conv1d(in_channels=FILTER_SIZE_3, out_channels=FILTER_SIZE_4, kernel_size=WIN_SIZE,
                                     stride=2, padding=WIN_SIZE // 2, dilation=1, groups=1, bias=True)
        self.dense_layer = torch.nn.Linear(in_features=FILTER_SIZE_4, out_features=DENSE_SIZE, bias=True)
        self.projection_layer = torch.nn.Linear(in_features=DENSE_SIZE, out_features=ntags, bias=True)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, words, word_chars):
        # char:
        batch_size, max_len, max_wlen = word_chars.shape
        ce = self.char_emb(word_chars)  # 2, 55, 11, 10
        ce = ce.permute(0, 1, 3, 2)
        c = self.cnn_char(ce.view(-1, CHAR_EMB_SIZE, max_wlen))
        c = c.view(batch_size, max_len, CHAR_CNN_FILTER_SIZE, max_wlen)
        c = c.max(dim=3)[0]
        c = F.relu(c)  # c is batch x nwords x filter_size
        c = c.permute(0, 2, 1)

        emb = self.emb(words)  # batch x nwords x emb_size
        emb = emb.permute(0, 2, 1)  # batch x emb_size x nwords

        combined_emb = torch.cat((emb, c), 1)
        combined_emb = combined_emb.permute(2, 0, 1)
        combined_emb, _ = self.gru_emb(
            combined_emb)  # improved embeddings, out: (seq_len, batch, num_directions * hidden_size)
        combined_emb = combined_emb.permute(1, 2, 0)  # batch x rnn_emb_size x nwords
        h = combined_emb

        h = self.conv3(h)  # batch x num_filters x nwords
        h = self.mpool3(h)
        h = F.relu(h)
        h = self.conv4(h)  # output is (batch, Channels, outseqlen)
        h = h.max(dim=2)[0]  # batch x num_filters
        h = F.relu(h)
        h = self.dense_layer(h)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.projection_layer(h)  # size(out) = batch x ntags
        return out


model = CNNClassifier()
print(model)
print("number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

train_samples = SentencesDataset(train_data)
dev_samples = SentencesDataset(dev_data)
train_dataloader_args = dict(shuffle=True, batch_size=256, pin_memory=True, collate_fn=collate_fn) if cuda \
    else dict(shuffle=False, batch_size=64, collate_fn=collate_fn)
train_loader = dataloader.DataLoader(train_samples, **train_dataloader_args)

test_dataloader_args = dict(shuffle=True, batch_size=64, pin_memory=True, collate_fn=collate_fn) if cuda \
    else dict(shuffle=False, batch_size=64, collate_fn=collate_fn)
dev_loader = dataloader.DataLoader(dev_samples, **test_dataloader_args)
# len(dev_loader)
len(train_samples), len(dev_samples)

Train_loss = []
Train_acc = []
Test_loss = []
Test_acc = []

n_epochs = 10
for i in range(n_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test_model(model, dev_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    Train_acc.append(train_acc)
    if len(Test_acc) > 1 and test_acc < max(Test_acc):
        print("val acc decreased")
    else:
        torch.save(model.state_dict(), "test_saved_model_edit")
    print('=' * 20)

model.load_state_dict(torch.load("test_saved_model_edit"))
test_model(model, dev_loader, criterion)

torch.save(model1.state_dict(), "test_saved_model_edit1")
torch.save(model2.state_dict(), "test_saved_model_edit2")
torch.save(model3.state_dict(), "test_saved_model_edit3")

import json

with open("vocw2i", 'w') as vf:
    json.dump(voc.w2i, vf)

test_model(model1, dev_loader, criterion)
test_model(model2, dev_loader, criterion)
test_model(model3, dev_loader, criterion)

test_data, test_topics = load_data(test_file)
test_samples = SentencesDataset(test_data)

predict_dataloader_args = dict(shuffle=False, batch_size=64, pin_memory=True, collate_fn=collate_fn) if cuda \
    else dict(shuffle=False, batch_size=64, collate_fn=collate_fn)
test_loader = dataloader.DataLoader(test_samples, **predict_dataloader_args)
dev_predict_loader = dataloader.DataLoader(dev_samples, **predict_dataloader_args)

model = model3

output = output_results(model, dev_predict_loader, criterion)
output_file = "valid_predictions_4.txt"
with open(output_file, 'w') as of:
    for i in range(len(output)):
        of.write("{}\n".format(idx_to_topic[output[i]]))

topic_to_idx['UNK'] = -1
output = output_results(model, test_loader, criterion)
output_file = "test_predictions_4.txt"
with open(output_file, 'w') as of:
    for i in range(len(output)):
        of.write("{}\n".format(idx_to_topic[output[i]]))


# !head valid_predictions.txt
# !wc valid_predictions.txt
# !wc topicclass/topicclass_valid.txt

# !head test_predictions.txt
# !wc test_predictions.txt
# !wc topicclass/topicclass_test.txt
# !head topicclass/topicclass_test.txt

def read_result(filename):
    out = []
    with open(filename) as f:
        for line in f:
            label = line.strip()
            out.append(label)
    return out


r1 = read_result("test_predictions_3.txt")
r2 = read_result("test_predictions_2.txt")
rbest = read_result("test_predictions.txt")

with open("ensemble_results_test.csv", "w") as wf:
    for i in range(len(rbest)):
        l1, l2, lb = r1[i], r2[i], rbest[i]
        if l1 == l2:
            l = l1
        else:
            l = lb
        wf.write("{}\n".format(l))

with open(dev_file) as df:
    with open("ensemble_results_valid.csv") as rf:
        correct = 0
        total = 0
        dfl = df.readlines()
        rfl = rf.readlines()
        for i in range(len(dfl)):
            if rfl[i].strip() == dfl[i].split('|||')[0].strip():
                correct += 1
            total += 1

correct, total, correct / total
