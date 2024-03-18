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


if __name__ == '__main__':
    datasets = 'music' # movie1m   movie20  music  book  restaurant  yelp
    kg_path = 'data/' + datasets + '/kg.txt'
    df_kg = pd.read_csv(kg_path, sep='\t', header=None, names=['head','relation','tail'])
    kg = _construct_kg(df_kg)
