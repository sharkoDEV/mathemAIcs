import argparse
import json
import os
import shutil
from typing import Dict, List

import matplotlib
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from model.multimodal_model import MultimodalModel
from model.tokenizer import SimpleTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class JsonlIndex:
    def __init__(self, path: str):
        self.path = path
        self.offsets: List[int] = []
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError(idx)
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)


class EntryDataset(Dataset):
    def __init__(self, manifest_path: str, text_path: str, base_dir: str):
        self.manifest = JsonlIndex(manifest_path)
        self.text = JsonlIndex(text_path)
        self.base_dir = base_dir

    def __len__(self) -> int:
        return len(self.manifest) + len(self.text)

    def __getitem__(self, idx: int) -> Dict:
        if idx < len(self.manifest):
            record = self.manifest.read(idx)
            image = os.path.join(self.base_dir, record["image"]) if record.get("image") else None
        else:
            record = self.text.read(idx - len(self.manifest))
            image = None
        return {
            "image": image,
            "prompt": record.get("prompt", ""),
            "answer": record["answer"],
        }


class MathDataset(Dataset):
    def __init__(
        self,
        entries,
        tokenizer: SimpleTokenizer,
        max_text_len: int = 128,
        image_size: int = 256,
    ) -> None:
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.entries[idx]
        if entry.get("image"):
            img = Image.open(entry["image"]).convert("RGB")
            image_tensor = self.transform(img)
            mask = 1.0
        else:
            image_tensor = torch.zeros(3, 256, 256)
            mask = 0.0
        prompt_ids = torch.tensor(
            self.tokenizer.encode(entry["prompt"], self.max_text_len), dtype=torch.long
        )
        target_ids = torch.tensor(
            self.tokenizer.encode(entry["answer"], self.max_text_len), dtype=torch.long
        )
        return {
            "image": image_tensor,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "image_mask": torch.tensor(mask, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([item["image"] for item in batch])
    prompt_ids = torch.stack([item["prompt_ids"] for item in batch])
    target_ids = torch.stack([item["target_ids"] for item in batch])
    image_mask = torch.stack([item["image_mask"] for item in batch])
    return {
        "images": images,
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "image_mask": image_mask,
    }


def plot_losses(train_losses: List[float], val_losses: List[float], output_path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch["images"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        mask = batch["image_mask"].to(device)
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]
        optimizer.zero_grad()
        logits = model(images, decoder_input, prompt_ids=prompt_ids, image_mask=mask)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), decoder_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask = batch["image_mask"].to(device)
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
            logits = model(images, decoder_input, prompt_ids=prompt_ids, image_mask=mask)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), decoder_target.reshape(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal math model")
    parser.add_argument("--manifest", default=os.path.join("dataset", "annotations", "dataset_manifest.jsonl"))
    parser.add_argument("--text", default=os.path.join("dataset", "text", "math_text.jsonl"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_file", default=os.path.join("models", "active.mathai"))
    parser.add_argument("--save_every", type=int, default=20, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    entries_dataset = EntryDataset(args.manifest, args.text, project_root)
    if len(entries_dataset) == 0:
        raise RuntimeError("No dataset entries found. Run build_dataset.py first.")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_dir = os.path.dirname(os.path.abspath(args.model_file))
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(
        entries_dataset[idx]["prompt"] + entries_dataset[idx]["answer"] for idx in range(len(entries_dataset))
    )

    full_dataset = MathDataset(entries_dataset, tokenizer, max_text_len=args.max_text_len)
    val_size = max(1, int(len(full_dataset) * 0.1))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultimodalModel(
        vocab_size=tokenizer.vocab_size,
        max_text_len=args.max_text_len,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    best_val_loss = float("inf")
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, tokenizer)
        val_loss = evaluate(model, val_loader, criterion, args.device, tokenizer)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "tokenizer": tokenizer.to_config(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "max_text_len": args.max_text_len,
                },
            }
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save(checkpoint, best_path)
            shutil.copyfile(best_path, args.model_file)
            print(f"Saved new best checkpoint and updated {args.model_file}.")
        if epoch % max(1, args.save_every) == 0 or epoch == args.epochs:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "tokenizer": tokenizer.to_config(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "max_text_len": args.max_text_len,
                },
            }
            ckpt_name = f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, ckpt_name))
            print(f"Saved periodic checkpoint {ckpt_name}.")

    plot_losses(train_losses, val_losses, os.path.join(args.checkpoint_dir, "loss.png"))


if __name__ == "__main__":
    main()
