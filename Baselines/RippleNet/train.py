import tensorflow as tf
import numpy as np
from model import RippleNet

def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]
    model = RippleNet(args, n_entity, n_relation)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc, train_f1,train_pre,train_reacll = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc, eval_f1,eval_pre,eval_reacll = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_acc, test_f1,test_pre,test_reacll = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)

            print('epoch %d    train auc: %.4f  acc: %.4f  f1:%.4f pre:%.4f recall:%.4f  eval auc: %.4f  acc: %.4f  f1:%.4f pre:%.4f recall:%.4f   test auc: %.4f  acc: %.4f  f1:%.4f pre:%.4f recall:%.4f '
                  % (step, train_auc, train_acc, train_f1,train_pre,train_reacll, eval_auc, eval_acc, eval_f1,eval_pre,eval_reacll, test_auc, test_acc, test_f1,test_pre, test_reacll))
def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    pre_list = []
    reacll_list = []
    while start < data.shape[0]:
        auc, acc, f1, pre, reacll = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        pre_list.append(pre)
        reacll_list.append(reacll)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list)), float(np.mean(pre_list)), float(np.mean(reacll_list))

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