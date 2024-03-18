from src.MKR import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Movie-20m
    # parser.add_argument('--dataset', type=str, default='movie')
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=5)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=1)
    # parser.add_argument('--T', type=int, default=5)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    # Movie-1m
    # parser.add_argument('--dataset', type=str, default='ml')
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=3)
    # parser.add_argument('--T', type=int, default=10)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    # Restaurant
    # parser.add_argument('--dataset', type=str, default='restaurant')
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=1)
    # parser.add_argument('--T', type=int, default=5)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    # Music
    # parser.add_argument('--dataset', type=str, default='music')
    # parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--l2', type=float, default=1e-3)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=3)
    # parser.add_argument('--T', type=int, default=10)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    # Yelp
    # parser.add_argument('--dataset', type=str, default='yelp')
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=1)
    # parser.add_argument('--T', type=int, default=5)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    # Book
    # parser.add_argument('--dataset', type=str, default='book')
    # parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--l2', type=float, default=1e-4)
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--epochs', type=int, default=30)
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dim', type=int, default=32)
    # parser.add_argument('--L', type=int, default=3)
    # parser.add_argument('--T', type=int, default=10)
    # parser.add_argument('--l1', type=float, default=1e-6)
    # parser.add_argument('--ratio', type=float, default=1)
    # parser.add_argument('--topk', type=float, default=10)

    args = parser.parse_args()

    train(args, False)

