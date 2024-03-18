import pandas as pd
from model import KGIE
from Dataset import KGIEDataset
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import random
from datetime import datetime

def train (args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    uAndR = data_loader.load_user_relations_interaction()
    iAndU = data_loader.load_item_users_interaction()
    x_train = data_loader.load_x_train()
    x_test = data_loader.load_x_test()
    x_validation = data_loader.x_validation
    num_user, num_entity, num_relation = data_loader.get_num()

    iAndU_len = len(iAndU)
    for k, v in tqdm(iAndU.items(), total=iAndU_len, desc='Organize items and number of users'):
        v_len = len(v)
        if (v_len < args.neighbor_sample_size):
            v.extend(random.choices(v, k=args.neighbor_sample_size - v_len))
            iAndU[k] = v
        elif (v_len > args.neighbor_sample_size):
            v = random.sample(v, args.neighbor_sample_size)
            iAndU[k] = v

    transit =[]
    transit2 = []
    uAndR_len = len(uAndR)

    for k,v in tqdm(uAndR.items(), total=uAndR_len, desc='Remove duplicate relations'):
        relations = pd.value_counts(uAndR[k]).index
        relations_len = len(relations)
        if(relations_len < args.neighbor_sample_size):
            for i in range(args.neighbor_sample_size - relations_len):
                transit2 = relations.tolist()
                transit.append(relations.values[0])
                transit2.extend(transit)
            transit = transit2
            transit2 = []
        else:
            transit = relations.tolist()[0:args.neighbor_sample_size]
        uAndR[k] = transit
        transit = []

    train_dataset = KGIEDataset(x_train)
    validation_dataset = KGIEDataset(x_validation)
    test_dataset = KGIEDataset(x_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    net = KGIE(num_user, num_entity, num_relation, kg, args, device, uAndR, iAndU).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)


    auc_score_list = []
    f1_score_list = []
    precision_score_list = []
    recall_score_list = []

    auc_score_t_list = []
    f1_score_t_list = []
    precision_score_t_list = []
    recall_t_score_list = []

    train_loader_len = len(train_loader)
    validation_loader_len = len(validation_loader)
    test_loader_len = len(test_loader)
    # train
    for epoch in range(args.n_epochs):
        net.train()
        for i, (user_ids, item_ids, labels) in tqdm(enumerate(train_loader), total=train_loader_len, desc='train'):
            user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids).to(device)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()

        # evaluate
        with torch.no_grad():
            total_roc = 0
            total_f1 = 0
            total_precision = 0
            total_recall = 0
            net.eval()
            for user_ids, item_ids, labels in tqdm(validation_loader, total=validation_loader_len,desc='validate'):
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids).to(device)
                pre_label = [1 if o.item() >= 0.5 else 0 for o in outputs]
                pre_label = torch.Tensor(pre_label)
                try:
                    total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    total_f1 += f1_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_precision += precision_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_recall += recall_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                except ValueError:
                    pass
            auc_score_list.append(total_roc / len(validation_loader))
            f1_score_list.append(total_f1/ len(validation_loader))
            precision_score_list.append(total_precision/len(validation_loader))
            recall_score_list.append(total_recall/len(validation_loader))

        # test
        with torch.no_grad():
            total_t_roc = 0
            total_t_f1 = 0
            total_t_precision = 0
            total_t_recall = 0
            net.eval()
            for user_ids, item_ids, labels in tqdm(test_loader, total=test_loader_len,desc='test'):
                user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
                outputs = net(user_ids, item_ids).to(device)
                pre_label = [1 if o.item() >= 0.5 else 0 for o in outputs]
                pre_label = torch.Tensor(pre_label)
                try:
                    total_t_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    total_t_f1 += f1_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_t_precision += precision_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                    total_t_recall += recall_score(labels.cpu().detach().numpy(), pre_label.cpu().detach().numpy(), average='binary')
                except ValueError:
                    pass
            auc_score_t_list.append(total_t_roc / len(test_loader))
            f1_score_t_list.append(total_t_f1 / len(test_loader))
            precision_score_t_list.append(total_t_precision / len(test_loader))
            recall_t_score_list.append(total_t_recall / len(test_loader))
        print( '\n'+ datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              '[Epoch {0}] '.format(epoch + 1),
              'Val auc:{:.4f}'.format(total_roc / len(validation_loader)),
              'f1:{:.4f}'.format(total_f1 / len(validation_loader)),
              'pre:{:.4f}'.format(total_precision / len(validation_loader)),
              'recall:{:.4f}'.format(total_recall / len(validation_loader)),
              'Test auc:{:.4f}'.format(total_t_roc / len(test_loader)),
              'f1:{:.4f}'.format(total_t_f1 / len(test_loader)),
              'pre:{:.4f}'.format(total_t_precision / len(test_loader)),
              'recall:{:.4f}'.format(total_t_recall / len(test_loader))
              )





