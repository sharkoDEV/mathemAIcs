from typing import Dict, Iterable, List


class SimpleTokenizer:
    def __init__(self) -> None:
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        self.itos: List[str] = list(self.special_tokens)

    def build_vocab(self, texts: Iterable[str]) -> None:
        for text in texts:
            for ch in text:
                if ch not in self.stoi:
                    self.stoi[ch] = len(self.itos)
                    self.itos.append(ch)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = [self.bos_id]
        for ch in text:
            tokens.append(self.stoi.get(ch, self.unk_id))
            if len(tokens) >= max_length - 1:
                break
        tokens.append(self.eos_id)
        if len(tokens) < max_length:
            tokens.extend([self.pad_id] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
            tokens[-1] = self.eos_id
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        chars: List[str] = []
        for tid in token_ids:
            if tid in (self.bos_id, self.pad_id):
                continue
            if tid == self.eos_id:
                break
            chars.append(self.itos[tid] if tid < len(self.itos) else "")
        return "".join(chars)

    def to_config(self) -> Dict:
        return {"itos": self.itos}

    @classmethod
    def from_config(cls, config: Dict) -> "SimpleTokenizer":
        tok = cls()
        tok.itos = config["itos"]
        tok.stoi = {token: idx for idx, token in enumerate(tok.itos)}
        return tok
