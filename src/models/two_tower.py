import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TwoTowerDataset(Dataset):
    def __init__(self, user_pos_items, n_items, samples_per_epoch):
        self.n_items = n_items
        self.samples_per_epoch = samples_per_epoch
        self.users = [u for u, items in user_pos_items.items() if len(items) > 0]
        self.user_pos = {u: list(items) for u, items in user_pos_items.items() if len(items) > 0}
        self.user_pos_set = {u: items for u, items in user_pos_items.items() if len(items) > 0}

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        u = self.users[np.random.randint(len(self.users))]
        pos = self.user_pos[u][np.random.randint(len(self.user_pos[u]))]
        neg = np.random.randint(self.n_items)
        while neg in self.user_pos_set[u]:
            neg = np.random.randint(self.n_items)
        return u, pos, neg


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, item_feat_dim,
                 emb_dim=32, tower_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(emb_dim, tower_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(tower_dim, tower_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(emb_dim + item_feat_dim, tower_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(tower_dim, tower_dim)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_user_repr(self, user):
        return self.user_tower(self.user_emb(user))

    def get_item_repr(self, item, item_feat):
        return self.item_tower(torch.cat([self.item_emb(item), item_feat], dim=-1))

    def forward(self, user, item, item_feat):
        return (self.get_user_repr(user) * self.get_item_repr(item, item_feat)).sum(dim=-1)


class TwoTowerWrapper:
    def __init__(self, model, n_users, n_items, item_features, device):
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self._precompute_items(item_features)

    def _precompute_items(self, item_features):
        self.model.eval()
        with torch.no_grad():
            items = torch.arange(self.n_items).long().to(self.device)
            reprs = []
            bs = 4096
            for i in range(0, self.n_items, bs):
                end = min(i + bs, self.n_items)
                r = self.model.get_item_repr(items[i:end], item_features[i:end])
                reprs.append(r.cpu().numpy())
            self.item_reprs = np.concatenate(reprs, axis=0)

    def predict_for_user(self, user_id, k=10, train_df=None):
        if user_id >= self.n_users:
            return []
        self.model.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_id]).to(self.device)
            u_repr = self.model.get_user_repr(u).cpu().numpy().flatten()

        scores = self.item_reprs @ u_repr

        if train_df is not None:
            seen = train_df[train_df['user_id'] == user_id]['item_id'].values
            scores[seen] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(int(i), float(scores[i])) for i in top_k if scores[i] > -np.inf]


def train_two_tower(model, dataset, device, item_features, n_epochs=30,
                    lr=0.001, reg=1e-5, batch_size=2048, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
    model.to(device)
    history = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for user, pos, neg in loader:
            user = user.long().to(device)
            pos = pos.long().to(device)
            neg = neg.long().to(device)
            pos_feat = item_features[pos]
            neg_feat = item_features[neg]

            pos_score = model(user, pos, pos_feat)
            neg_score = model(user, neg, neg_feat)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        history.append(avg)
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: loss = {avg:.4f}")

    if verbose:
        print(f"Epoch {n_epochs - 1:3d}: loss = {history[-1]:.4f}")
    return history