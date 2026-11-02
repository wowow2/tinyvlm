import torch

def load_tokens(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)

def train_val_split(dataset: torch.Tensor, val_frac: float):
    n = int((1 - val_frac) * len(dataset))
    train_set, val_set = dataset[:n], dataset[n:]
    return train_set, val_set

class TokenDataset:
    def __init__(self, tokens: torch.Tensor, block_size: int, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokens = tokens.to(device)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens)

    def get_example(self, i: int):
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]
        return x, y

    def get_batch(self, batch_size: int, device: str = None):
        n = len(self.tokens) - self.block_size
        ix = torch.randint(n, (batch_size,), device=self.tokens.device)
        idx = ix[:, None] + torch.arange(self.block_size, device=self.tokens.device)
        x = self.tokens[idx]
        y = self.tokens[idx + 1]
        if device is not None and str(device) != str(self.tokens.device):
            return x.to(device), y.to(device)
        return x, y

def make_datasets(text: str, tokenizer, block_size: int, val_frac: float = 0.1, device: str = None):
    tokens = load_tokens(text, tokenizer)
    train, val = train_val_split(tokens, val_frac)
    return TokenDataset(train, block_size, device), TokenDataset(val, block_size, device)