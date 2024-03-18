import time
import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score
from torch import optim
import random
from src.evaluate import get_hit, get_ndcg
from src.load_base import load_data, get_records


class CrossAndCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossAndCompressUnit, self).__init__()
        self.dim = dim
        self.w_vv = nn.Parameter(t.randn(dim, 1))
        self.w_ev = nn.Parameter(t.randn(dim, 1))
        self.w_ve = nn.Parameter(t.randn(dim, 1))
        self.w_vv = nn.Parameter(t.randn(dim, 1))
        self.b_v = nn.Parameter(t.randn(dim, 1))
        self.b_e = nn.Parameter(t.randn(dim, 1))

    def forward(self, v, e):
        C = t.matmul(e.view(-1, self.dim, 1), v)
        v = t.matmul(C, self.w_vv) + t.matmul(C, self.w_ev) + self.b_v
        e = t.matmul(C, self.w_ve) + t.matmul(C, self.w_vv) + self.b_e
        return v.view(-1, 1, self.dim), e.view(-1, 1, self.dim)


class CAC1(nn.Module):
    def __init__(self, dim):
        super(CAC1, self).__init__()
        self.cac1 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        return v, e


class CAC2(nn.Module):
    def __init__(self, dim):
        super(CAC2, self).__init__()
        self.cac1 = CrossAndCompressUnit(dim)
        self.cac2 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        v, e = self.cac2(v, e)
        return v, e


class CAC3(nn.Module):
    def __init__(self, dim):
        super(CAC3, self).__init__()
        self.cac1 = CrossAndCompressUnit(dim)
        self.cac2 = CrossAndCompressUnit(dim)
        self.cac3 = CrossAndCompressUnit(dim)

    def forward(self, v, e):
        v, e = self.cac1(v, e)
        v, e = self.cac2(v, e)
        v, e = self.cac3(v, e)
        return v, e


class MLP1(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP1, self).__init__()
        self.l1 = nn.Linear(int_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        return y


class MLP2(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP2, self).__init__()
        self.l1 = nn.Linear(int_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        y = t.relu(self.l2(y))
        return y


class MLP3(nn.Module):
    def __init__(self, int_dim, hidden_dim, out_dim):
        super(MLP3, self).__init__()
        self.l1 = nn.Linear(int_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        y = t.relu(self.l1(x))
        y = t.relu(self.l2(y))
        y = t.relu(self.l3(y))
        return y


class MKR(nn.Module):

    def __init__(self, dim, L, T, l1, n_entities, n_user, n_item, n_relations):
        super(MKR, self).__init__()
        self.dim = dim
        self.L = L
        self.T = T
        self.l1 = l1
        self.M_k = nn.Linear(2*dim, dim)
        if L == 1:
            self.user_mlp = MLP1(dim, dim, dim)
            self.r_mlp = MLP1(dim, dim, dim)
            self.tail_mlp = MLP1(dim, dim, dim)
            self.cac = CAC1(dim)
        elif L == 2:
            self.user_mlp = MLP2(dim, dim, dim)
            self.r_mlp = MLP2(dim, dim, dim)
            self.tail_mlp = MLP2(dim, dim, dim)
            self.cac = CAC2(dim)
        else:
            self.user_mlp = MLP3(dim, dim, dim)
            self.r_mlp = MLP3(dim, dim, dim)
            self.tail_mlp = MLP3(dim, dim, dim)
            self.cac = CAC3(dim)

        rs_entity_embedding = t.randn(n_entities, dim)
        rs_item_embedding = t.randn(n_item, dim)
        e_entity_embedding = t.randn(n_entities, dim)
        e_item_embedding = t.randn(n_item, dim)
        relation_embedding = t.randn(n_relations, dim)
        user_embedding_matrix = t.rand(n_user, dim)

        nn.init.xavier_uniform_(rs_entity_embedding)
        nn.init.xavier_uniform_(rs_item_embedding)
        nn.init.xavier_uniform_(e_entity_embedding)
        nn.init.xavier_uniform_(e_item_embedding)
        nn.init.xavier_uniform_(relation_embedding)
        nn.init.xavier_uniform_(user_embedding_matrix)

        self.rs_entity_embedding = nn.Parameter(rs_entity_embedding)
        self.rs_item_embedding = nn.Parameter(rs_item_embedding)
        self.e_entity_embedding = nn.Parameter(e_entity_embedding)
        self.e_item_embedding = nn.Parameter(e_item_embedding)
        self.relation_embedding = nn.Parameter(relation_embedding)
        self.user_embedding_matrix = nn.Parameter(user_embedding_matrix)
        self.criterion = nn.BCELoss()

    def forward(self, data):
        users = self.user_embedding_matrix[[i[0] for i in data]]
        items = self.rs_item_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        item_entities = self.e_entity_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        u_L = self.user_mlp(users)
        v_L = self.cac(items, item_entities)[0].view(-1, self.dim)
        predicts = ((u_L * v_L).sum(dim=1)).view(-1)
        return predicts

    def cal_kg_loss(self, data):

        heads = self.e_entity_embedding[[i[0] for i in data]].view(-1, 1, self.dim)
        relations = self.relation_embedding[[i[1] for i in data]]
        pos_tails = self.e_entity_embedding[[i[2] for i in data]]
        pos_tails = self.tail_mlp(pos_tails)
        neg_tails = self.e_entity_embedding[[i[3] for i in data]]
        neg_tails = self.tail_mlp(neg_tails)
        items = self.rs_item_embedding[[i[0] for i in data]].view(-1, 1, self.dim)

        true_scores = self.get_kg_scores(heads, relations, pos_tails, items)
        false_scores = self.get_kg_scores(heads, relations, neg_tails, items)

        return -self.l1 * (true_scores - false_scores)

    def get_kg_scores(self, heads, relations, tails, items):
        h_L = self.cac(items, heads)[1].view(-1, self.dim)
        r_L = self.r_mlp(relations)

        pred_tails = t.relu(self.M_k(t.cat([h_L, r_L], dim=1)))

        scores = ((pred_tails * tails).sum(dim=1)).sum()

        return scores

    def cal_rs_loss(self, data):
        users = self.user_embedding_matrix[[i[0] for i in data]]
        items = self.rs_item_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        item_entities = self.e_entity_embedding[[i[1] for i in data]].view(-1, 1, self.dim)
        labels = t.tensor([float(i[2]) for i in data]).view(-1)
        if t.cuda.is_available():
            labels = labels.to(users.device)

        u_L = self.user_mlp(users)
        v_L = self.cac(items, item_entities)[0].view(-1, self.dim)

        predicts = t.sigmoid(((u_L * v_L).sum(dim=1)).view(-1))
        return self.criterion(predicts, labels)

def eval_topk(model, rec):
    precision_list = []
    model.eval()
    for user in (rec):
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = model.forward(pairs)
        n = len(pairs)
        item_scores = {items[i]: predict[i] for i in range(n)}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())

        precision_list.append([len({items[-1]}.intersection(item_list[: k])) / k for k in [1, 2, 3, 4, 5, 10, 20]])

    model.train()
    return np.array(precision_list).mean(axis=0).tolist()


def eval_ctr(model, pairs, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model(pairs[i: i+batch_size]).cpu().detach().numpy().tolist()
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    acc = f1_score(true_label, pred_label)
    return auc, acc


def get_hrtts(kg_dict, n_item):

    entities = list(kg_dict)

    hrtts = []
    for head in range(n_item):
        for r_t in kg_dict[head]:
            relation = r_t[0]
            positive_tail = r_t[1]

            while True:
                negative_tail = np.random.choice(entities, 1)[0]
                if [relation, negative_tail] not in kg_dict[head]:
                    hrtts.append([head, relation, positive_tail, negative_tail])
                    break
    np.random.shuffle(hrtts)
    return hrtts


def train(args, is_topk=False):
    np.random.seed(2022)
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, rec, kg_dict = data[4], data[5], data[6], data[7], data[8]
    hrtts = get_hrtts(kg_dict, n_item)
    model = MKR(args.dim, args.L, args.T, args.l1, n_entity, n_user, n_item, n_relation)
    if t.cuda.is_available():
        model = model.to(args.device)
    print(args.dataset + '-----------------------------------')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('l1: %1.0e' % args.l1, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    train_auc_list = []
    train_acc_list = []
    eval_auc_list = []
    eval_acc_list = []
    test_auc_list = []
    test_acc_list = []
    HR_list = []
    NDCG_list = []
    precision_list = []

    for epoch in (range(args.epochs)):
        start = time.perf_counter()
        model.train()
        loss_sum = 0

        size = len(hrtts)
        np.random.shuffle(hrtts)
        for i in range(0, size, args.batch_size):
            next_i = min([size, i + args.batch_size])
            data = hrtts[i: next_i]
            loss = model.cal_kg_loss(data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        size = len(train_set)
        np.random.shuffle(train_set)
        for j in range(model.T):
            for i in range(0, size, args.batch_size):
                next_i = min([size, i + args.batch_size])
                data = train_set[i: next_i]
                loss = model.cal_rs_loss(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

        train_auc, train_acc = eval_ctr(model, train_set, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, args.batch_size)

        print('epoch: %d \t train_auc: %.4f \t train_f1: %.4f \t '
              'eval_auc: %.4f \t eval_f1: %.4f \t test_auc: %.4f \t test_f1: %.4f \t' %
              ((epoch + 1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        HR, NDCG = 0, 0
        precisions = []
        if is_topk:
            precisions = eval_topk(model, rec)

            print("Precision: ", end='[')

            for i in range(len(precisions)):

                if i == len(precisions) - 1:
                    print('%.4f' % precisions[i], end='] ')
                else:
                    print('%.4f' % precisions[i], end=', ')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)
        precision_list.append(precisions)
        end = time.perf_counter()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.4f \t train_acc: %.4f \t eval_auc: %.4f \t eval_acc: %.4f \t '
          'test_auc: %.4f \t test_acc: %.4f \t' %
          (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
           test_auc_list[indices], test_acc_list[indices]), end='\t')
    print("Precision: ", end='')
    print(precision_list[indices])
    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]




