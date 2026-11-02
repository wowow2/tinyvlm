import torch

def load_tokens(text: str, tokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)

def train_val_split(dataset: torch.Tensor, val_frac: float):
    n = int((1-val_frac) * len(dataset))
    train_set, val_set = dataset[:n], dataset[n:]
    return train_set, val_set

class TokenDataset:
    def __init__(self, tokens: torch.Tensor, block_size: int, ):
        self.tokens = tokens.to(device)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens)

    def get_example(self, i: int):
        x = self.tokens[i: i + self.block_size] # block size is context window
        y = self.tokens[i + 1: i + self.block_size + 1]
        return x, y

    def get_batch(self, batch_size: int):
        ix = torch.randint(len(self.tokens) - self.block_size, (batch_size,), device=self.tokens.device)
        x = torch.stack([self.tokens[i: i + self.block_size] for i in ix])
        y = torch.stack([self.tokens[i + 1: i + self.block_size + 1] for i in ix])
        return x, y

def make_datasets(text: str, tokenizer, block_size: int, val_frac: float = 0.1):
    """
    Singleton constructor for datasets
    """
    tokens = load_tokens(text, tokenizer)
    train, val = train_val_split(tokens, val_frac)
    return TokenDataset(train, block_size), TokenDataset(val, block_size)