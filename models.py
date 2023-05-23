import torch
from torch import nn
import torch.nn.functional as F

class MatrixFactorizationMachine(nn.Module):
    def __init__(self, n_users, n_movies, embd_dim, features_dim, dropout=0.4):
        super().__init__()
        self.user_embd = nn.Embedding(n_users, embd_dim)
        self.movie_embd = nn.Embedding(n_movies, embd_dim)
        self.ln = nn.Linear(features_dim + 2 * embd_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idxs):
        movie_emb = self.movie_embd(idxs[:,0])
        user_emb = self.user_embd(idxs[:,1])
        out = torch.cat([user_emb, movie_emb, x], dim=1)
        out = self.dropout(out)
        out = self.ln(out)
        return out
    

class CandidatesGenerator(nn.Module):
    def __init__(self, 
                 n_movies,
                 movie_emb_dim,    
                 sparse_matrix_dim,
                 hidden_dim):
        super().__init__()
        # embedding/lookup table for movies
        self.movie_embd = nn.Embedding(n_movies+1, movie_emb_dim, padding_idx=n_movies)

        # linear layers
        self.ln1 = nn.Linear(movie_emb_dim + sparse_matrix_dim, hidden_dim * 4)
        self.ln2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        # classifier layers
        self.classifier = nn.Linear(hidden_dim * 2, n_movies)

        # dropout
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, Xd, Xw):
        # get average vectors of watched movies
        watched_embedding = torch.mean(self.movie_embd(Xw.long()), axis=1)
        # concatenate watched movies' vector to user features 
        out = torch.cat([Xd, watched_embedding], axis=1)

        out = self.ln1(out)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.dropout(out)

        out = self.ln2(out)
        logits = F.leaky_relu(out, negative_slope=0.2)
        logits = self.dropout(logits)

        logits = self.classifier(logits)
        return out, logits