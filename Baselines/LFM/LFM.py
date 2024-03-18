import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class LFM:
    def __init__(self, num_factors, learning_rate, reg_param, num_iterations, eval_interval):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.num_iterations = num_iterations
        self.eval_interval = eval_interval
        self.val_aucs = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.test_aucs = []
        self.test_f1s = []
        self.test_precisions = []
        self.test_recalls = []

    def fit(self, X_train, X_val, X_test):
        num_users = int(np.max(X_train[:, 0].astype(int))) + 1
        num_items = int(np.max(X_train[:, 1].astype(int))) + 1

        self.user_factors = np.random.normal(size=(num_users, self.num_factors))
        self.item_factors = np.random.normal(size=(num_items, self.num_factors))

        best_auc = 0.0
        best_val_auc = 0.0

        for iteration in tqdm(range(self.num_iterations)):
            for user_id, item_id, rating in X_train:
                prediction = np.dot(self.user_factors[int(user_id)], self.item_factors[int(item_id)])
                gradient_u = 2 * (prediction - float(rating)) * self.item_factors[int(item_id)]
                gradient_i = 2 * (prediction - float(rating)) * self.user_factors[int(user_id)]
                self.user_factors[int(user_id)] -= self.learning_rate * (gradient_u + self.reg_param * self.user_factors[int(user_id)])
                self.item_factors[int(item_id)] -= self.learning_rate * (gradient_i + self.reg_param * self.item_factors[int(item_id)])

            if iteration % self.eval_interval == 0:
                val_auc, val_f1, val_precision, val_recall = self.evaluate(X_val)
                print("Iteration:", iteration)
                print("Validation AUC:", val_auc)
                print("Validation F1:", val_f1)
                print("Validation Precision:", val_precision)
                print("Validation Recall:", val_recall)

                self.val_aucs.append(val_auc)
                self.val_f1s.append(val_f1)
                self.val_precisions.append(val_precision)
                self.val_recalls.append(val_recall)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_iteration = iteration

            test_auc, test_f1, test_precision, test_recall = self.evaluate(X_test)
            print("Test AUC:", test_auc)
            print("Test F1:", test_f1)
            print("Test Precision:", test_precision)
            print("Test Recall:", test_recall)

            self.test_aucs.append(test_auc)
            self.test_f1s.append(test_f1)
            self.test_precisions.append(test_precision)
            self.test_recalls.append(test_recall)

            if test_auc > best_auc:
                best_auc = test_auc
                self.best_user_factors = self.user_factors.copy()
                self.best_item_factors = self.item_factors.copy()

        self.user_factors = self.best_user_factors
        self.item_factors = self.best_item_factors

        best_val_auc, best_val_f1, best_val_precision, best_val_recall = self.evaluate(X_val)
        print("Best Validation AUC:", best_val_auc)
        print("Best Validation F1:", best_val_f1)
        print("Best Validation Precision:", best_val_precision)
        print("Best Validation Recall:", best_val_recall)

        best_test_auc, best_test_f1, best_test_precision, best_test_recall = self.evaluate(X_test)
        print("Best Test AUC:", best_test_auc)
        print("Best Test F1:", best_test_f1)
        print("Best Test Precision:", best_test_precision)
        print("Best Test Recall:", best_test_recall)

    def predict(self, user_ids, item_ids):
        return np.dot(self.user_factors[user_ids], self.item_factors[item_ids])

    def evaluate(self, X, threshold=0.5):
        y_true = []
        y_pred = []

        for user_id, item_id, rating in X:
            y_true.append(int(rating))
            y_pred.append(self.predict(int(user_id), int(item_id)))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred > threshold)
        pre = precision_score(y_true, y_pred > threshold)
        recall = recall_score(y_true, y_pred > threshold)

        return auc, f1, pre, recall


def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
    train, middle = train_test_split(data, test_size=ratio_test + ratio_val, random_state=2022)
    ratio = ratio_val / (ratio_test + ratio_val)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation

if __name__ == '__main__':

    df_dataset = pd.read_csv("data/movie1m_ratings.txt", sep='\t', dtype=int)
    x_train, x_test, x_validation = train_test_val_split(df_dataset, 0.6, 0.2, 0.2)
    X_train = x_train.to_numpy()
    X_val = x_validation.to_numpy()
    X_test = x_test.to_numpy()

    model = LFM(num_factors=10, learning_rate=0.04, reg_param=0.01, num_iterations=10000, eval_interval=500)
    model.fit(X_train, X_val, X_test)

    best_val_auc = max(model.val_aucs)
    best_val_f1 = model.val_f1s[model.val_aucs.index(best_val_auc)]
    best_val_precision = model.val_precisions[model.val_aucs.index(best_val_auc)]
    best_val_recall = model.val_recalls[model.val_aucs.index(best_val_auc)]
    print("Best Validation AUC:", best_val_auc)
    print("Best Validation F1:", best_val_f1)
    print("Best Validation Precision:", best_val_precision)
    print("Best Validation Recall:", best_val_recall)
    best_test_auc = max(model.test_aucs)
    best_test_f1 = model.test_f1s[model.test_aucs.index(best_test_auc)]
    best_test_precision = model.test_precisions[model.test_aucs.index(best_test_auc)]
    best_test_recall = model.test_recalls[model.test_aucs.index(best_test_auc)]
    print("Best Test AUC:", best_test_auc)
    print("Best Test F1:", best_test_f1)
    print("Best Test Precision:", best_test_precision)
    print("Best Test Recall:", best_test_recall)
