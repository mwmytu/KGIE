import time
import torch as t
import torch.nn as nn
import numpy as np
from torch import optim
import random
from src.evaluate import get_hit, get_ndcg
from src.load_base import load_data
from sklearn.metrics import roc_auc_score, f1_score

class CKE(nn.Module):

    def __init__(self, n_entity, n_user, n_item, n_rels, dim):

        super(CKE, self).__init__()
        self.dim = dim
        user_emb_matrix = t.randn(n_user, dim)
        item_emb_matrix = t.randn(n_item, dim)
        ent_emb_matrix = t.randn(n_entity, dim)
        Mr_matrix = t.randn(n_rels, dim, dim)
        rel_emb_matrix = t.randn(n_rels, dim)

        nn.init.xavier_uniform_(user_emb_matrix)
        nn.init.xavier_uniform_(item_emb_matrix)
        nn.init.xavier_uniform_(ent_emb_matrix)
        nn.init.xavier_uniform_(Mr_matrix)
        nn.init.xavier_uniform_(rel_emb_matrix)

        self.user_emb_matrix = nn.Parameter(user_emb_matrix)
        self.item_emb_matrix = nn.Parameter(item_emb_matrix)
        self.ent_emb_matrix = nn.Parameter(ent_emb_matrix)
        self.Mr_matrix = nn.Parameter(Mr_matrix)
        self.rel_emb_matrix = nn.Parameter(rel_emb_matrix)

    def forward(self, data, name):
        if name == 'kg':
            heads_id = [i[0] for i in data]
            relations_id = [i[1] for i in data]
            pos_tails_id = [i[2] for i in data]
            neg_tails_id = [i[3] for i in data]
            head_emb = self.ent_emb_matrix[heads_id].view(-1, 1, self.dim)
            rel_emb = self.rel_emb_matrix[relations_id].view(-1, 1, self.dim)
            pos_tail_emb = self.ent_emb_matrix[pos_tails_id].view(-1, 1, self.dim)
            neg_tail_emb = self.ent_emb_matrix[neg_tails_id].view(-1, 1, self.dim)
            Mr = self.Mr_matrix[relations_id]
            pos_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(pos_tail_emb, Mr)).norm(dim=[1, 2])
            neg_stru_scores = (t.matmul(head_emb, Mr) + rel_emb - t.matmul(neg_tail_emb, Mr)).norm(dim=[1, 2])
            stru_loss = t.sigmoid(pos_stru_scores - neg_stru_scores)
            stru_loss = t.log(stru_loss).sum()
            return stru_loss
        else:
            users_id = [i[0] for i in data]
            poss_id = [i[1] for i in data]
            negs_id = [i[2] for i in data]
            users_emb = self.user_emb_matrix[users_id]
            pos_items_emb = []
            for id in poss_id:
                if id < len(self.item_emb_matrix) and id < len(self.ent_emb_matrix):
                    pos_items_emb.append(self.item_emb_matrix[id] + self.ent_emb_matrix[id])
                else:
                    random_index = random.randint(0, min(len(self.item_emb_matrix), len(self.ent_emb_matrix)) - 1)
                    pos_items_emb.append(self.item_emb_matrix[random_index] + self.ent_emb_matrix[random_index])
            pos_items_emb = t.stack(pos_items_emb)

            neg_items_emb = []
            for id in negs_id:
                if id < len(self.item_emb_matrix) and id < len(self.ent_emb_matrix):
                    neg_items_emb.append(self.item_emb_matrix[id] + self.ent_emb_matrix[id])
                else:
                    random_index = random.randint(0, min(len(self.item_emb_matrix), len(self.ent_emb_matrix)) - 1)
                    neg_items_emb.append(self.item_emb_matrix[random_index] + self.ent_emb_matrix[random_index])
            neg_items_emb = t.stack(neg_items_emb)
            base_loss = t.sigmoid(t.mul(users_emb, pos_items_emb).sum(dim=1) - t.mul(users_emb, neg_items_emb).sum(dim=1))
            base_loss = t.log(base_loss).sum()
            return base_loss

    def get_predict(self, pairs):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]
        user_emb = self.user_emb_matrix[users]
        item_emb = []
        for item in items:
            if item < len(self.item_emb_matrix) and item < len(self.ent_emb_matrix):
                item_emb.append(self.item_emb_matrix[item] + self.ent_emb_matrix[item])
            else:
                random_index = random.randint(0, min(len(self.item_emb_matrix), len(self.ent_emb_matrix)) - 1)
                item_emb.append(self.item_emb_matrix[random_index] + self.ent_emb_matrix[random_index])
        item_emb = t.stack(item_emb)
        score = (user_emb * item_emb).sum(dim=1)
        return score.cpu().detach().view(-1).numpy().tolist()


def eval_ctr(model, pairs, batch_size):

    model.eval()
    pred_label = []
    for i in range(0, len(pairs), batch_size):
        batch_label = model.get_predict(pairs[i: i+batch_size])
        pred_label.extend(batch_label)
    model.train()

    true_label = [pair[2] for pair in pairs]
    auc = roc_auc_score(true_label, pred_label)

    pred_np = np.array(pred_label)
    pred_np[pred_np >= 0.5] = 1
    pred_np[pred_np < 0.5] = 0
    pred_label = pred_np.tolist()
    f1 = f1_score(true_label, pred_label)
    return auc, f1

def eval_topk(model, rec):
    precision_list = []
    model.eval()
    for user in rec:
        items = list(rec[user])
        pairs = [[user, item] for item in items]
        predict = []
        predict.extend(model.get_predict(pairs))
        n = len(pairs)
        item_scores = {items[i]: predict[i] for i in range(n)}
        item_list = list(dict(sorted(item_scores.items(), key=lambda x: x[1], reverse=True)).keys())
        precision_list.append([len({items[-1]}.intersection(item_list[:k])) / k for k in [1, 2, 3, 4, 5, 10, 20]])
    model.train()
    return np.array(precision_list).mean(axis=0).tolist()


def get_uvvs(pairs):
    positive_dict = {}
    negative_dict = {}
    for pair in pairs:
        user = pair[0]
        item = pair[1]
        label = pair[2]
        if label == 1:
            if user not in positive_dict:
                positive_dict[user] = []

            positive_dict[user].append(item)
        else:
            if user not in negative_dict:
                negative_dict[user] = []

            negative_dict[user].append(item)
    data = []
    for user in positive_dict:
        size = len(positive_dict[user])
        for i in range(size):
            pos_item = positive_dict[user][i]
            neg_item = negative_dict[user][i]
            data.append([user, pos_item, neg_item])
    np.random.shuffle(data)
    return data


def get_uvvs2(pairs):
    positive_dict = {}
    negative_dict = {}
    neg_item = []
    for pair in pairs:
        user = pair[0]
        item = pair[1]
        label = pair[2]
        if label == 1:
            if user not in positive_dict:
                positive_dict[user] = []

            positive_dict[user].append(item)
        else:
            if user not in negative_dict:
                negative_dict[user] = []

            negative_dict[user].append(item)
    data = []
    for user in positive_dict:
        size = 0
        n_size = 0
        p_size = len(positive_dict[user])
        if user in negative_dict:
            n_size = len(negative_dict[user])
            if p_size > n_size:
                size = n_size
            else:
                size = p_size
        else:
            n_size = 0
        for i in range(size):
            pos_item = positive_dict[user][i]
            if n_size == 0:
                neg_item = 0
            else:
                neg_item = negative_dict[user][i]
            data.append([user, pos_item, neg_item])
            neg_item = []

    np.random.shuffle(data)
    return data

def get_hrtts(kg_dict):
    entities = list(kg_dict)
    hrtts = []
    for head in kg_dict:
        for r_t in kg_dict[head]:
            relation = r_t[0]
            positive_tail = r_t[1]
            while True:
                negative_tail = np.random.choice(entities, 1)[0]
                if [relation, negative_tail] not in kg_dict[head]:
                    hrtts.append([head, relation, positive_tail, negative_tail])
                    break
    return hrtts


def train(args, is_topk=False):
    np.random.seed(555)

    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, eval_set, test_set, kg_dict = data[4], data[5], data[6], data[7]
    rec = data[8]
    hrtts = get_hrtts(kg_dict)
    model = CKE(n_entity, n_user, n_item, n_relation, args.dim)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    uvvs = get_uvvs(train_set)
    train_data = [hrtts, uvvs]
    print(args.dataset + '----------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
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

    for epoch in range(args.epochs):

        start = time.perf_counter()
        size = len(train_data[0])
        start_index = 0
        loss_sum = 0
        np.random.shuffle(train_set)
        np.random.shuffle(hrtts)
        while start_index < size:
            if start_index + args.batch_size <= size:
                hrtts = train_data[0][start_index: start_index + args.batch_size]
            else:
                hrtts = train_data[0][start_index:]
            loss = -model(hrtts, 'kg')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        start_index = 0
        size = len(train_data[-1])
        while start_index < size:
            if start_index + args.batch_size <= size:
                uvvs = train_data[-1][start_index: start_index + args.batch_size]
            else:
                uvvs = train_data[-1][start_index:]
            loss = -model(uvvs, 'cf')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()

            start_index += args.batch_size

        train_auc, train_acc = eval_ctr(model, train_set, args.batch_size)
        eval_auc, eval_acc = eval_ctr(model, eval_set, args.batch_size)
        test_auc, test_acc = eval_ctr(model, test_set, args.batch_size)

        print('epoch: %d \t train_auc: %.4f \t train_f1: %.4f \t '
              'eval_auc: %.4f \t eval_f1: %.4f \t test_auc: %.4f \t test_f1: %.4f \t' %
              ((epoch+1), train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc), end='\t')

        HR, NDCG = 0, 0
        precision = []
        if is_topk:
            # HR, NDCG = eval_topk(model, rec, args.topk)
            # print('HR: %.4f NDCG: %.4f' % (HR, NDCG), end='\t')

            precision = eval_topk(model, rec)
            print("Precision: [", end='')

            for i in range(len(precision)):

                if i == len(precision) - 1:
                    print('%.4f' % precision[i], end=']')
                else:
                    print('%.4f' % precision[i], end=', ')

        train_auc_list.append(train_auc)
        train_acc_list.append(train_acc)
        eval_auc_list.append(eval_auc)
        eval_acc_list.append(eval_acc)
        test_auc_list.append(test_auc)
        test_acc_list.append(test_acc)
        HR_list.append(HR)
        NDCG_list.append(NDCG)
        precision_list.append(precision)
        end = time.perf_counter()
        print('time: %d' % (end - start))

    indices = eval_auc_list.index(max(eval_auc_list))
    print(args.dataset, end='\t')
    print('train_auc: %.4f \t train_acc: %.4f \t eval_auc: %.4f \t eval_acc: %.4f \t '
          'test_auc: %.4f \t test_acc: %.4f \t' %
        (train_auc_list[indices], train_acc_list[indices], eval_auc_list[indices], eval_acc_list[indices],
         test_auc_list[indices], test_acc_list[indices]), end='\t')
    print('Precision: ', end='')
    print(precision_list[indices])
    return eval_auc_list[indices], eval_acc_list[indices], test_auc_list[indices], test_acc_list[indices]


