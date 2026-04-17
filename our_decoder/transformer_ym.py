import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
from preprocessing_ym import load_dataset, preprocess

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, n_features, n_outputs, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.2):
        super().__init__()
        # project features vector to dimension of model
        self.input_proj = nn.Linear(n_features, d_model)

        # all features processed in parallel so model needs positional encodings to know what comes first
        self.pos_enc = PositionalEncoding(d_model)

        # zeros out 20% of features randomly during training and scales rest
        # forces model to not rely heavily on one set of features vs another
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu" # apparently gelu works better for relu on small models
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # collapses vectors into size of time step dimension

        return self.output_head(x)

class TransformerDecoder:
    def __init__(self, d_model=128, n_heads=4, n_layers=2, d_ff=256,
                 dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=64,
                 num_epochs=50, patience=20, verbose=True):
        self.d_model = d_model # model dimension
        self.n_heads = n_heads # number of self attention heads
        self.n_layers = n_layers # more layers more capacity to learn complex patterns, but overfit
        self.d_ff = d_ff # hidden size of feedforward network inside each encode layer
        self.dropout = dropout # randomly zeros out 20% of features
        self.lr = lr # learning rate for optimizer
        self.weight_decay = weight_decay # shrinks weights each step prevents overfitting stops weights growing too large
        self.batch_size = batch_size # how many samples are processed in one forward pass
        self.num_epochs = num_epochs # max number of full passes through data
        self.patience = patience # if validation loss does not improve for # of patience epochs stops, saves weights
        self.verbose = verbose
        self.device = torch.device('mps')
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        n_features = X_train.shape[2]
        n_outputs = y_train.shape[1]

        self.model = Transformer(
            n_features, n_outputs, self.d_model, self.n_heads,
            self.n_layers, self.d_ff, self.dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )
        criterion = nn.MSELoss()

        # Build dataloaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        use_valid = X_valid is not None and y_valid is not None
        if use_valid:
            X_val_t = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_valid, dtype=torch.float32).to(self.device)

        # Training loop with early stopping
        best_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            scheduler.step()
            train_loss /= len(train_ds)

            # Validation
            if use_valid:
                self.model.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model(X_val_t), y_val_t).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs} — "
                          f"train: {train_loss:.5f}  val: {val_loss:.5f}  "
                          f"lr: {scheduler.get_last_lr()[0]:.2e}")
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # if self.verbose and (epoch + 1) % 10 == 0:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.num_epochs} — train: {train_loss:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, X_test):
        self.model.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()
        return preds

X, y = load_dataset('../data/example_data_m1.pickle')
data_splits = preprocess(X, y)


decoder = TransformerDecoder()
decoder.fit(data_splits["X_train"], data_splits["y_train"])
y_pred = decoder.predict(data_splits["X_test"])

from Neural_Decoding.metrics import get_R2, get_rho

R2 = get_R2(data_splits["y_test"], y_pred)
rho = get_rho(data_splits["y_test"], y_pred)

print(f"R2:  {R2}")
print(f"rho: {rho}")