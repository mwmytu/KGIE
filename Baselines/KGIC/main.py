import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()

# Movie-20m
# parser.add_argument('-d', '--dataset', type=str, default='movie20')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--n_layer', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.004)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=32)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='sum')

# Movie-1m
# parser.add_argument('-d', '--dataset', type=str, default='movie1m')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--n_layer', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.004)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=32)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='concat')

# Restaurant
# parser.add_argument('-d', '--dataset', type=str, default='restaurant')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--n_layer', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.004)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=32)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='concat')

# Music
# parser.add_argument('-d', '--dataset', type=str, default='music')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--n_layer', type=int, default=1)
# parser.add_argument('--lr', type=float, default=0.002)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=16)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='sum')

# Yelp
# parser.add_argument('-d', '--dataset', type=str, default='yelp')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--n_layer', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.004)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=32)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='concat')

# Book
# parser.add_argument('-d', '--dataset', type=str, default='book')
# parser.add_argument('--n_epoch', type=int, default=20)
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--n_layer', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.004)
# parser.add_argument('--l2_weight', type=float, default=1e-4)
# parser.add_argument('--dim', type=int, default=32)
# parser.add_argument('--user_triple_set_size', type=int, default=40)
# parser.add_argument('--user_potential_triple_set_size', type=int, default=128)
# parser.add_argument('--item_origin_triple_set_size', type=int, default=40)
# parser.add_argument('--item_triple_set_size', type=int, default=128)
# parser.add_argument('--agg', type=str, default='concat')

parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--show_topk', type=bool, default=False)
parser.add_argument('--random_flag', type=bool, default=False)
args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(304, 2018)

data_info = load_data(args)
train(args, data_info)
