from typing import Optional

import torch
from torch import nn

from .text_decoder import TextDecoder
from .vision_encoder import VisionEncoder


class MultimodalModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_text_len: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.context_dim = d_model
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_len=max_text_len,
        )
        self.vision_encoder = VisionEncoder(output_dim=d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.max_text_len = max_text_len

    def build_context(
        self,
        images: Optional[torch.Tensor],
        prompt_ids: Optional[torch.Tensor],
        image_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prompt_ids is None and images is None:
            raise ValueError("At least one modality must be provided")
        device = None
        if prompt_ids is not None:
            device = prompt_ids.device
        elif images is not None:
            device = images.device
        batch_size = prompt_ids.size(0) if prompt_ids is not None else images.size(0)
        context = torch.zeros(batch_size, self.context_dim, device=device)
        if prompt_ids is not None:
            prompt_embed = self.text_decoder.token_embedding(prompt_ids)
            prompt_context = prompt_embed.mean(dim=1)
            context = context + prompt_context
        if images is not None:
            vision_context = self.vision_encoder(images)
            if image_mask is not None:
                vision_context = vision_context * image_mask.unsqueeze(1)
            context = context + vision_context
        context = self.context_norm(context)
        return context

    def forward(
        self,
        images: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = self.build_context(images, prompt_ids, image_mask)
        logits = self.text_decoder(decoder_input_ids, context)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt_ids: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
        image_mask: Optional[torch.Tensor],
        max_new_tokens: int = 64,
    ) -> str:
        device = next(self.parameters()).device
        seq = torch.full((1, 1), tokenizer.bos_id, dtype=torch.long, device=device)
        if prompt_ids is not None:
            prompt_ids = prompt_ids.to(device)
        if image_tensor is not None:
            image_tensor = image_tensor.to(device)
        context = self.build_context(image_tensor, prompt_ids, image_mask)
        output_tokens = []
        for _ in range(max_new_tokens):
            logits = self.text_decoder(seq, context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            token_id = next_token.item()
            if token_id == tokenizer.eos_id:
                break
            output_tokens.append(token_id)
            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
        return tokenizer.decode(output_tokens)
