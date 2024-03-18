import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

parser = argparse.ArgumentParser()

# Movie-20m
# parser.add_argument('-d', '--dataset', type=str, default='movie')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-5)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='concat')

# Movie-1m
# parser.add_argument('-d', '--dataset', type=str, default='movie1m')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-5)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='concat')

# Restaurant
# parser.add_argument('-d', '--dataset', type=str, default='restaurant')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-5)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='concat')

# Music
# parser.add_argument('-d', '--dataset', type=str, default='music')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-3)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='pool')

# Book
# parser.add_argument('-d', '--dataset', type=str, default='book')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-3)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='pool')

# Yelp
# parser.add_argument('-d', '--dataset', type=str, default='yelp')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=4096)
# parser.add_argument('--n_layer', type=int, default=3)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-3)
# parser.add_argument('--dim', type=int, default=64)
# parser.add_argument('--user_triple_set_size', type=int, default=8)
# parser.add_argument('--item_triple_set_size', type=int, default=64)
# parser.add_argument('--agg', type=str, default='pool')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(304, 2024)
    
data_info = load_data(args)
train(args, data_info)