import collections
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile

# 下载数据集
# d2l.download_imdb('E:\\program\\python\\SentimentAnalysis\\data\\aclImdb')
# 读取训练数据和测试数据集
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
# 创建词典
vocab = d2l.get_vocab_imdb(train_data)

# 创建数据迭代器
batch_size = 64
train_set = gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle = True)
test_iter = gdata.DataLoader(train_set, batch_size, shuffle = True)

# # 创建含两个隐藏层的双向循环神经网络
# embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
# RNN_net = d2l.BiRNN(vocab, embed_size, num_hiddens, num_layers)
# RNN_net.initialize(init.Xavier(), ctx = ctx)
# #RNN网络加载预训练的词向量，为词典中的vocab中的每个词先加载100维的GloVe词向量
# glove_embedding = text.embedding.create(
#     'glove', pretrained_file_name = 'glove.6B.100d.txt', vocabulary = vocab)
# net.embedding.weight.set_data(glove_embedding.idx_to_vec)
# net.embedding.collect_params().setattr('grad_req', 'null')

#创建一个TextCNN，三个卷积层，核分别为3， 4， 5，输出通道数为100
embed_size, kernel_size, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
CNN_net = d2l.TextCNN(vocab, embed_size, kernel_size, nums_channels)
CNN_net.initialize(init.Xavier(), ctx = ctx)
#CNN网络加载预训练的词向量，为词典中的vocab中的每个词先加载100维的GloVe词向量
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name = 'glove.6B.100d.txt', vocabulary = vocab)
CNN_net.embedding.weight.set_data(glove_embedding.idx_to_vec)
CNN_net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
CNN_net.constant_embedding.collect_params().setattr('grad_req', 'null')

#训练网络
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(CNN_net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, CNN_net, loss, trainer, ctx, num_epochs)
