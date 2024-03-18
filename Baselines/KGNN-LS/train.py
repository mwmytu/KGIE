import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from model import KGNN_LS



def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    interaction_table, offset = get_interaction_table(train_data, n_entity)
    model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.initializer.run()

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1,train_pre,train_recall = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1,eval_pre,eval_recall = ctr_eval(sess, model, eval_data, args.batch_size)
            test_auc, test_f1,test_pre,test_recall = ctr_eval(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  f1: %.4f pre: %.4f recall: %.4f    eval auc: %.4f  f1: %.4f pre: %.4f recall: %.4f   test auc: %.4f  f1: %.4f pre: %.4f recall: %.4f '
                  % (step, train_auc, train_f1,train_pre,train_recall, eval_auc, eval_f1,eval_pre,eval_recall, test_auc, test_f1,test_pre,test_recall))
            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')


def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)
    interaction_table = tf.lookup.StaticHashTable(  
        tf.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    pre_list = []
    reacll_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1,pre,reacll = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        pre_list.append(pre)
        reacll_list.append(reacll)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)),float(np.mean(pre_list)),float(np.mean(reacll_list))


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def create_data(combined_list):
    result = []
    for user, items in combined_list:
        for item in items:
            if item == items[0]:
                result.append([user, item, 1])
            else:
                result.append([user, item, 0])

    nparray = np.array(result)
    return nparray