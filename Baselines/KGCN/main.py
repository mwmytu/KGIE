import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()

# Movie-20m
# parser.add_argument('--dataset', type=str, default='movie20', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# Movie-1m
# parser.add_argument('--dataset', type=str, default='movie1m', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# Restaurant
# parser.add_argument('--dataset', type=str, default='restaurant', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=2, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of l2 regularization') # 1e-7
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate') # 2e-2
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# Music
# parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size') # 32
# parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# Yelp
# parser.add_argument('--dataset', type=str, default='yelp', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=60, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')

# Book
# parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
# parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
# parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
# parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
# parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
# parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--l2_weight', type=float, default=2e-6, help='weight of l2 regularization')
# parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
# parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')


show_loss = False
show_time = False
show_topk = False

t = time()

args = parser.parse_args()
data = load_data(args)
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
