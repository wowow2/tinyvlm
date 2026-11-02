class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self._build_lookups()

    @classmethod
    def from_chars(cls, chars):
        obj = cls.__new__(cls)
        obj.chars = list(chars)
        obj._build_lookups()
        return obj

    def _build_lookups(self):
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[c] for c in tokens)
