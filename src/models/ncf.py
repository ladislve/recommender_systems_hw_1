import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BPRTripletDataset(Dataset):
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


class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, mlp_dims=None):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [64, 32]

        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        layers = []
        in_dim = emb_dim * 2
        for dim in mlp_dims:
            layers.extend([nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(0.1)])
            in_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(emb_dim + mlp_dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(emb.weight, 0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user, item):
        gmf = self.gmf_user(user) * self.gmf_item(item)
        mlp_in = torch.cat([self.mlp_user(user), self.mlp_item(item)], dim=-1)
        mlp = self.mlp(mlp_in)
        return self.output(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)


class NeuMFWrapper:
    def __init__(self, model, n_users, n_items, device):
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.device = device

    def predict_for_user(self, user_id, k=10, train_df=None):
        if user_id >= self.n_users:
            return []
        self.model.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_id]).to(self.device)
            items = torch.arange(self.n_items).long().to(self.device)
            u_exp = u.expand(self.n_items)
            scores = np.zeros(self.n_items)
            bs = 4096
            for i in range(0, self.n_items, bs):
                end = min(i + bs, self.n_items)
                s = self.model(u_exp[i:end], items[i:end])
                scores[i:end] = s.cpu().numpy()

        if train_df is not None:
            seen = train_df[train_df['user_id'] == user_id]['item_id'].values
            scores[seen] = -np.inf

        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(int(i), float(scores[i])) for i in top_k if scores[i] > -np.inf]


def train_neumf(model, dataset, device, n_epochs=30, lr=0.001, reg=1e-5,
                batch_size=2048, verbose=True):
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

            loss = -torch.log(torch.sigmoid(model(user, pos) - model(user, neg)) + 1e-10).mean()
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