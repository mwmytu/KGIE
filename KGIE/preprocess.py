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

def _construct_interactive_embedding(x_train, kg, datasets):
    uAndR = dict()
    iAndU = dict()
    df_len = len(x_train['userID'])
    for index, row in tqdm(x_train.iterrows(), total=df_len, desc='Obtaining interation'):
        userID = row['userID']
        itemID = row['itemID']
        label = row['label']
        if (label == 1):
            if (itemID) in iAndU:
                iAndU[itemID].append(userID)
            else:
                iAndU[itemID] = [userID]

            for num in kg.get(itemID):
                relation = num[0]
                if userID in uAndR:
                    uAndR[userID].append(relation)
                else:
                    uAndR[userID] = [relation]

    with open('data/'+datasets+'/user_relations_interaction.pickle', 'wb') as f:
        pickle.dump(uAndR, f)
    with open('data/'+datasets+'/item_users_interactive.pickle', 'wb') as f:
        pickle.dump(iAndU, f)



if __name__ == '__main__':
    datasets = 'yelp' # movie1m   movie20  music  book  restaurant  yelp
    kg_path = 'data/' + datasets + '/kg.txt'
    train_path = 'data/' + datasets + '/train.txt'
    df_kg = pd.read_csv(kg_path, sep='\t', header=None, names=['head','relation','tail'])
    x_train = pd.read_csv(train_path, sep='\t')
    kg = _construct_kg(df_kg)
    _construct_interactive_embedding(x_train, kg, datasets)
