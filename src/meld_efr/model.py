
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel


class DialogueMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_emotions: int,
        max_speakers: int = 16,
        speaker_dim: int = 32,
        context_hidden: int = 256,
        dropout: float = 0.25,
        encoder_config_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_emotions = num_emotions
        self.max_speakers = max_speakers
        self.speaker_dim = speaker_dim
        self.context_hidden = context_hidden
        self.dropout = dropout

        if encoder_config_dir is None:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            encoder_config = AutoConfig.from_pretrained(
                encoder_config_dir, local_files_only=True
            )
            self.encoder = AutoModel.from_config(encoder_config)
        encoder_hidden = self.encoder.config.hidden_size

        self.speaker_embedding = nn.Embedding(
            max_speakers + 1, speaker_dim, padding_idx=0
        )
        self.context_gru = nn.GRU(
            input_size=encoder_hidden + speaker_dim,
            hidden_size=context_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        contextual_dim = context_hidden * 2
        self.context_norm = nn.LayerNorm(contextual_dim)

        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(contextual_dim, contextual_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(contextual_dim // 2, num_emotions),
        )
        self.trigger_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(contextual_dim + num_emotions, contextual_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(contextual_dim // 2, 1),
        )

    def architecture_config(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "num_emotions": self.num_emotions,
            "max_speakers": self.max_speakers,
            "speaker_dim": self.speaker_dim,
            "context_hidden": self.context_hidden,
            "dropout": self.dropout,
        }

    def encode_utterances(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_u, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(-1, seq_len)
        flat_attn = attention_mask.reshape(-1, seq_len)
        flat_valid = utterance_mask.reshape(-1)
        valid_idx = flat_valid.nonzero(as_tuple=False).squeeze(-1)

        if valid_idx.numel() == 0:
            raise ValueError("Batch senza utterance valide.")

        valid_ids = flat_ids.index_select(0, valid_idx)
        valid_attn = flat_attn.index_select(0, valid_idx)
        encoder_out = self.encoder(
            input_ids=valid_ids, attention_mask=valid_attn
        ).last_hidden_state[:, 0]

        hidden = encoder_out.shape[-1]
        flat_repr = encoder_out.new_zeros((batch_size * max_u, hidden))
        flat_repr.index_copy_(0, valid_idx, encoder_out)
        return flat_repr.view(batch_size, max_u, hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        utterance_mask: torch.Tensor,
        speaker_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        utter_repr = self.encode_utterances(input_ids, attention_mask, utterance_mask)
        speaker_repr = self.speaker_embedding(speaker_ids)
        x = torch.cat([utter_repr, speaker_repr], dim=-1)

        lengths = utterance_mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_context, _ = self.context_gru(packed)
        context, _ = pad_packed_sequence(
            packed_context, batch_first=True, total_length=x.shape[1]
        )
        context = self.context_norm(context)

        emotion_logits = self.emotion_head(context)
        # Il detach evita che il task trigger trascini l'emotion classifier verso scorciatoie.
        emotion_probs = torch.softmax(emotion_logits, dim=-1).detach()
        trigger_logits = self.trigger_head(
            torch.cat([context, emotion_probs], dim=-1)
        ).squeeze(-1)
        return emotion_logits, trigger_logits
