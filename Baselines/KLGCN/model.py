import torch
import random
from tqdm import tqdm
from aggregator import Aggregator

class KLGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device, UAndI, iAndU):
        super(KLGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        self._gen_adj()
        self.usr = torch.nn.Embedding(num_user, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        self.iAndU = iAndU
        self.UAndI = UAndI

    def _gen_adj(self):
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        for e in tqdm(self.kg, total=len(self.kg)):
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, u, v):

        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        item_neighbor = torch.empty(self.batch_size, self.n_neighbor, dtype=torch.long).to(self.device)
        user_neighbor = torch.empty(self.batch_size, self.n_neighbor, dtype=torch.long).to(self.device)
        ki = 0
        for user in u:
            if user.item() in self.UAndI:
                user_neighbor[ki] = torch.LongTensor(self.UAndI[user.item()])
            else:
                user_neighbor[ki] = torch.LongTensor([0] * self.n_neighbor)
            ki = ki+1
        q = 0
        for items in v:
            if items.item() in self.iAndU:
                item_neighbor[q] = torch.LongTensor(self.iAndU[items.item()])
            else:
                item_neighbor[q] = torch.LongTensor([0] * self.n_neighbor)
            q = q+1
        item_neighbor_embeddings = self.usr(item_neighbor)
        user_neighbor_embeddings = self.ent(user_neighbor)
        item_neighbor_embeddings = torch.mul(item_neighbor_embeddings, 1 / self.n_neighbor)
        user_neighbor_embeddings = torch.mul(user_neighbor_embeddings, 1 / self.n_neighbor)
        lite_user_embeddings = torch.sum(item_neighbor_embeddings, dim=1)
        lite_item_embeddings = torch.sum(user_neighbor_embeddings, dim=1)
        u = u.view((-1, 1))
        v = v.view((-1, 1))
        user_embeddings = self.usr(u).squeeze(dim = 1)
        item_embeddings_original = self.ent(v).squeeze(dim = 1)
        entities, relations = self._get_neighbors(v)
        item_embeddings = self._aggregate(user_embeddings, entities, relations, item_embeddings_original)
        user_embeddings = (0.5*lite_user_embeddings)+(0.5*user_embeddings)
        item_embeddings = (0.5*lite_item_embeddings)+(0.5*item_embeddings)

        scores = (user_embeddings * item_embeddings).sum(dim = 1)
        return torch.sigmoid(scores)
    
    def _get_neighbors(self, v):

        entities = [v]
        relations = []
        neighbor_entities = torch.LongTensor(self.adj_ent[entities[0].cpu()]).view((self.batch_size, -1)).to(self.device)
        neighbor_relations = torch.LongTensor(self.adj_rel[entities[0].cpu()]).view((self.batch_size, -1)).to(self.device)
        entities.append(neighbor_entities)
        relations.append(neighbor_relations)
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations, item_embeddings_originl):
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        vector = self.aggregator(
            self_vectors=entity_vectors[0],
            neighbor_vectors=entity_vectors[1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
            neighbor_relations=relation_vectors[0].view((self.batch_size, -1, self.n_neighbor, self.dim)),
            user_embeddings=user_embeddings,
            item_embeddings_originl = item_embeddings_originl)
        return vector


