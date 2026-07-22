
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import DialogueCollator, DialogueDataset, load_records
from .model import DialogueMultiTaskModel
from .utils import get_device, write_json


def load_model_bundle(checkpoint_path: str | Path, device_name: str = "auto"):
    device = get_device(device_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    required = {
        "model_state_dict",
        "architecture",
        "emotion_to_id",
        "trigger_threshold",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint incompleto: campi mancanti {sorted(missing)}")

    architecture = dict(checkpoint["architecture"])
    artifact_dir = Path(checkpoint_path).resolve().parent
    encoder_config_dir = artifact_dir / "encoder_config"
    if not encoder_config_dir.exists():
        raise FileNotFoundError(
            f"Config encoder locale non trovata: {encoder_config_dir}. "
            "Conserva la cartella artifacts/ completa insieme al checkpoint."
        )
    model = DialogueMultiTaskModel(
        **architecture, encoder_config_dir=str(encoder_config_dir)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer_dir = artifact_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, use_fast=True, local_files_only=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(architecture["model_name"], use_fast=True)
    return model, tokenizer, checkpoint, device


@torch.no_grad()
def predict_records(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path | None = None,
    device_name: str = "auto",
) -> list[dict[str, Any]]:
    model, tokenizer, checkpoint, device = load_model_bundle(checkpoint_path, device_name)
    records = load_records(input_path, require_labels=False)
    architecture = checkpoint["architecture"]
    training_config = checkpoint.get("training_config", {})
    max_tokens = int(training_config.get("max_utterance_tokens", 64))
    collator = DialogueCollator(
        tokenizer,
        emotion_to_id=None,
        max_utterance_tokens=max_tokens,
        max_speakers=int(architecture.get("max_speakers", 16)),
        include_labels=False,
    )
    loader = DataLoader(
        DialogueDataset(records), batch_size=1, shuffle=False, collate_fn=collator
    )
    id_to_emotion = {int(idx): label for label, idx in checkpoint["emotion_to_id"].items()}
    threshold = float(checkpoint["trigger_threshold"])

    results: list[dict[str, Any]] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        utterance_mask = batch["utterance_mask"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        emotion_logits, trigger_logits = model(
            input_ids, attention_mask, utterance_mask, speaker_ids
        )
        record = batch["records"][0]
        n = len(record["utterances"])
        emotion_ids = emotion_logits[0, :n].argmax(dim=-1).cpu().tolist()
        trigger_prob = torch.sigmoid(trigger_logits[0, :n]).cpu().tolist()
        predictions = []
        for i in range(n):
            predictions.append(
                {
                    "index": i,
                    "speaker": record["speakers"][i],
                    "utterance": record["utterances"][i],
                    "emotion": id_to_emotion[int(emotion_ids[i])],
                    "trigger_probability": round(float(trigger_prob[i]), 6),
                    "trigger": int(trigger_prob[i] >= threshold),
                }
            )
        results.append(
            {
                "trigger_threshold": threshold,
                "predictions": predictions,
            }
        )

    if output_path is not None:
        write_json(output_path, results)
    return results
