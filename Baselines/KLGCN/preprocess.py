import pandas as pd
from tqdm import tqdm
import pickle

def _construct_kg(df_kg):
    kg = dict()
    for i in tqdm(range(len(df_kg)), total=len(df_kg), desc='Creat Kg'):
        head = df_kg.iloc[i]['head']
        relation = df_kg.iloc[i]['relation']
        tail = df_kg.iloc[i]['tail']
        if head in kg:
            kg[head].append((relation, tail))
        else:
            kg[head] = [(relation, tail)]
        if tail in kg:
            kg[tail].append((relation, head))
        else:
            kg[tail] = [(relation, head)]
    with open('data/'+datasets+'/kg.pickle', 'wb') as f:
        pickle.dump(kg, f)
    return kg

def _construct_neighbor(x_train, datasets):
    UAndI = dict()
    df_len = len(x_train['userID'])
    for index, row in tqdm(x_train.iterrows(), total=df_len):
        userID = row['userID']
        itemID = row['itemID']
        label = row['label']
        if (label == 1):
            if (userID) in UAndI:
                UAndI[userID].append(itemID)
            else:
                UAndI[userID] = [itemID]
    iAndU = dict()
    df_len = len(x_train['userID'])
    for index, row in tqdm(x_train.iterrows(), total=df_len):
        userID = row['userID']
        itemID = row['itemID']
        label = row['label']
        if (label == 1):
            if (itemID) in iAndU:
                iAndU[itemID].append(userID)
            else:
                iAndU[itemID] = [userID]

    with open('data/'+datasets+'/U_neighbor.pickle', 'wb') as f:
        pickle.dump(UAndI, f)
    with open('data/'+datasets+'/I_neighbor.pickle', 'wb') as f:
        pickle.dump(iAndU, f)

if __name__ == '__main__':
    datasets = 'restaurant' # movie1m   movie20  music  book  restaurant  yelp
    kg_path = 'data/' + datasets + '/kg.txt'
    train_path = 'data/' + datasets + '/train.txt'
    df_kg = pd.read_csv(kg_path, sep='\t', header=None, names=['head','relation','tail'])
    x_train = pd.read_csv(train_path, sep='\t')
    _construct_kg(df_kg)
    _construct_neighbor(x_train, datasets)
