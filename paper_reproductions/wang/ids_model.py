"""IDS MLP model for MANDA.

MLP: input -> hidden(50, ReLU) -> output(2, softmax).
Per paper: one input layer, one hidden layer with 50 neurons, one output layer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IDSModel(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 50, n_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def predict_proba(self, x):
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


def train_ids(X_train, y_train, X_test, y_test, epochs=30, batch=256, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = X_train.shape[1]
    model = IDSModel(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xs = torch.from_numpy(X_test).to(device)
    ys = torch.from_numpy(y_test).to(device)

    n = Xt.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx]
            yb = yt[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xt).argmax(1) == yt).float().mean().item()
            te_acc = (model(Xs).argmax(1) == ys).float().mean().item()
        print(f"epoch {epoch+1:02d} | loss {total_loss/n:.4f} | train_acc {tr_acc:.4f} | test_acc {te_acc:.4f}")
    return model


if __name__ == "__main__":
    from data_loader import load_nsl_kdd, preprocess
    tr, te = load_nsl_kdd()
    d = preprocess(tr, te)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    model = train_ids(d["X_train"], d["y_train"], d["X_test"], d["y_test"], device=device)
    torch.save({"state_dict": model.state_dict(), "in_dim": d["n_features"]}, "ids_model.pt")
    print("saved ids_model.pt")
