import os
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

from model.multimodal_model import MultimodalModel
from model.tokenizer import SimpleTokenizer


class Predictor:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.tokenizer = SimpleTokenizer.from_config(checkpoint["tokenizer"])
        config = checkpoint.get("config", {})
        self.max_text_len = config.get("max_text_len", 128)
        self.model = MultimodalModel(
            vocab_size=self.tokenizer.vocab_size,
            max_text_len=self.max_text_len,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def _prepare_image(self, image: Optional[Image.Image]):
        if image is None:
            return None, None
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return tensor, torch.tensor([1.0], device=self.device)

    def predict(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 64,
    ) -> Dict:
        if not prompt and image is None:
            raise ValueError("Provide at least an image or a prompt.")
        prompt_ids = None
        if prompt:
            prompt_tensor = torch.tensor(
                self.tokenizer.encode(prompt, self.max_text_len), dtype=torch.long
            ).unsqueeze(0)
            prompt_ids = prompt_tensor.to(self.device)
        image_tensor, image_mask = self._prepare_image(image)
        answer = self.model.generate(
            tokenizer=self.tokenizer,
            prompt_ids=prompt_ids,
            image_tensor=image_tensor,
            image_mask=image_mask,
            max_new_tokens=max_new_tokens,
        )
        return {
            "prompt": prompt,
            "answer": answer,
            "has_image": image is not None,
        }


def load_predictor_from_env() -> Predictor:
    checkpoint_path = os.getenv("MODEL_FILE", os.path.join("models", "active.mathai"))
    return Predictor(checkpoint_path)
