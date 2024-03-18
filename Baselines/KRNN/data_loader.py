import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

class DataLoader:
    def __init__(self, data):
        self.cfg = {
            'movie20': {
                'item2id_path': 'data/movie20/item_index2entity_id.txt',
                'kg_path': 'data/movie20/kg.txt',
                'rating_path': 'data/movie20/ratings.txt',
                'rating_sep': '\t',
                'kgPath': 'data/movie20/kg.pickle',
                'trainPath': 'data/movie20/train.txt',
                'testPath': 'data/movie20/test.txt',
                'valPath': 'data/movie20/val.txt',
                'userList':'data/movie20/user_list.txt'
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/ratings.txt',
                'rating_sep': '\t',
                'kgPath':'data/music/kg.pickle',
                'trainPath':'data/music/train.txt',
                'testPath': 'data/music/test.txt',
                'valPath': 'data/music/val.txt',
                'userList': 'data/music/user_list.txt'

            },
            'book': {
                'item2id_path': 'data/book/item_index2entity_id.txt',
                'kg_path': 'data/book/kg.txt',
                'rating_path': 'data/book/ratings.txt',
                'rating_sep': '\t',
                'kgPath': 'data/book/kg.pickle',
                'trainPath': 'data/book/train.txt',
                'testPath': 'data/book/test.txt',
                'valPath': 'data/book/val.txt',
                'userList': 'data/book/user_list.txt'
            },
            'restaurant': {
                'item2id_path': 'data/restaurant/item_index2entity_id.txt',
                'kg_path': 'data/restaurant/kg.txt',
                'rating_path': 'data/restaurant/ratings.txt',
                'rating_sep': '\t',
                'kgPath': 'data/restaurant/kg.pickle',
                'trainPath': 'data/restaurant/train.txt',
                'testPath': 'data/restaurant/test.txt',
                'valPath': 'data/restaurant/val.txt',
                'userList': 'data/restaurant/user_list.txt'
            },
            'yelp': {
                'item2id_path': 'data/yelp/item_index2entity_id.txt',
                'kg_path': 'data/yelp/kg.txt',
                'rating_path': 'data/yelp/ratings.txt',
                'rating_sep': '\t',
                'kgPath': 'data/yelp/kg.pickle',
                'trainPath': 'data/yelp/train.txt',
                'testPath': 'data/yelp/test.txt',
                'valPath': 'data/yelp/val.txt',
                'userList': 'data/yelp/user_list.txt'
            },
            'movie1m': {
                'item2id_path': 'data/movie1m/item_index2entity_id.txt',
                'kg_path': 'data/movie1m/kg.txt',
                'rating_path': 'data/movie1m/ratings.txt',
                'rating_sep': '\t',
                'kgPath': 'data/movie1m/kg.pickle',
                'trainPath': 'data/movie1m/train.txt',
                'testPath': 'data/movie1m/test.txt',
                'valPath': 'data/movie1m/val.txt',
                'userList': 'data/movie1m/user_list.txt'
            },
        }
        self.data = data
        df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item', 'id'])
        df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
        with open(self.cfg[data]['kgPath'], 'rb') as f:
            kg = pickle.load(f)
        x_train = pd.read_csv(self.cfg[data]['trainPath'], sep='\t')
        x_test = pd.read_csv(self.cfg[data]['testPath'], sep='\t')
        x_validation = pd.read_csv(self.cfg[data]['valPath'], sep='\t')
        user_List = pd.read_csv(self.cfg[data]['userList'], sep='\t', header=None, names=['userID'])
        self.kg = kg
        self.x_train = x_train
        self.x_test = x_test
        self.x_validation = x_validation
        self.df_item2id = df_item2id
        self.df_kg = df_kg
        self.user_List = user_List['userID']
        self.user_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()
        self._encoding()

    def _encoding(self):
        self.user_encoder.fit(self.user_List)
        self.entity_encoder.fit(pd.concat([self.df_item2id['id'], self.df_kg['head'], self.df_kg['tail']]))
        self.relation_encoder.fit(self.df_kg['relation'])
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def load_kg(self):
        return self.kg

    def load_x_train(self):
        return self.x_train

    def load_x_validation(self):
        return self.x_validation

    def load_x_test(self):
        return self.x_test

    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))