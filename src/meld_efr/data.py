
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .utils import read_json


def _clean_trigger(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if np.isnan(value):
            return 0.0
    except (TypeError, ValueError):
        pass
    return float(value)


def validate_record(record: dict[str, Any], require_labels: bool = True) -> None:
    required = {"utterances", "speakers"}
    if require_labels:
        required |= {"emotions", "triggers"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Campi mancanti nel dialogo: {sorted(missing)}")

    n = len(record["utterances"])
    if n == 0:
        raise ValueError("Un dialogo deve contenere almeno una utterance.")
    if len(record["speakers"]) != n:
        raise ValueError("speakers e utterances devono avere la stessa lunghezza.")
    if require_labels:
        if len(record["emotions"]) != n or len(record["triggers"]) != n:
            raise ValueError(
                "utterances, speakers, emotions e triggers devono avere la stessa lunghezza."
            )


def normalize_record(
    record: dict[str, Any],
    require_labels: bool = True
) -> dict[str, Any]:

    out = dict(record)

    # MELD_test_efr.json usa "labels" invece di "triggers"
    if require_labels and "triggers" not in out and "labels" in out:
        out["triggers"] = out["labels"]

    validate_record(out, require_labels=require_labels)

    out["utterances"] = [str(x) for x in out["utterances"]]
    out["speakers"] = [str(x) for x in out["speakers"]]

    if require_labels:
        out["emotions"] = [str(x) for x in out["emotions"]]
        out["triggers"] = [
            _clean_trigger(x) for x in out["triggers"]
        ]

    return out

def load_records(path: str | Path, require_labels: bool = True) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Formato non valido in {path}: attesa lista di dialoghi o singolo dialogo.")
    return [normalize_record(x, require_labels=require_labels) for x in payload]


def split_train_val_test(
    records: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) < 10:
        raise ValueError(
            "Servono almeno 10 dialoghi per creare automaticamente train/val/test. "
            "In alternativa passa --val-path e --test-path."
        )
    strat = [int(any(t > 0.5 for t in r["triggers"])) for r in records]
    try:
        train, temp = train_test_split(
            records, test_size=0.20, random_state=seed, stratify=strat
        )
        temp_strat = [int(any(t > 0.5 for t in r["triggers"])) for r in temp]
        val, test = train_test_split(
            temp, test_size=0.50, random_state=seed, stratify=temp_strat
        )
    except ValueError:
        train, temp = train_test_split(records, test_size=0.20, random_state=seed)
        val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    return train, val, test


def flatten_field(records: Sequence[dict[str, Any]], field: str) -> list[Any]:
    return [item for record in records for item in record[field]]


def build_emotion_mapping(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    classes = sorted(set(flatten_field(records, "emotions")))
    if len(classes) < 2:
        raise ValueError("Il training set deve contenere almeno due classi di emozione.")
    return {label: idx for idx, label in enumerate(classes)}


def speakers_to_local_ids(speakers: Sequence[str], max_speakers: int) -> list[int]:
    mapping: dict[str, int] = {}
    ids: list[int] = []
    next_id = 1  # 0 = padding
    for speaker in speakers:
        if speaker not in mapping:
            mapping[speaker] = min(next_id, max_speakers)
            next_id += 1
        ids.append(mapping[speaker])
    return ids


class DialogueDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class DialogueCollator:
    def __init__(
        self,
        tokenizer: Any,
        emotion_to_id: dict[str, int] | None,
        max_utterance_tokens: int = 64,
        max_speakers: int = 16,
        include_labels: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.emotion_to_id = emotion_to_id
        self.max_utterance_tokens = max_utterance_tokens
        self.max_speakers = max_speakers
        self.include_labels = include_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_u = max(len(x["utterances"]) for x in batch)

        flat_texts: list[str] = []
        utterance_mask = torch.zeros(batch_size, max_u, dtype=torch.bool)
        speaker_ids = torch.zeros(batch_size, max_u, dtype=torch.long)

        if self.include_labels:
            if self.emotion_to_id is None:
                raise ValueError("emotion_to_id è obbligatorio quando include_labels=True")
            emotion_labels = torch.full((batch_size, max_u), -100, dtype=torch.long)
            trigger_labels = torch.zeros(batch_size, max_u, dtype=torch.float32)

        for b, record in enumerate(batch):
            n = len(record["utterances"])
            utterance_mask[b, :n] = True
            speaker_ids[b, :n] = torch.tensor(
                speakers_to_local_ids(record["speakers"], self.max_speakers), dtype=torch.long
            )
            flat_texts.extend(record["utterances"] + [""] * (max_u - n))

            if self.include_labels:
                try:
                    emotion_labels[b, :n] = torch.tensor(
                        [self.emotion_to_id[e] for e in record["emotions"]], dtype=torch.long
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Classe emotion {exc.args[0]!r} non presente nel training set."
                    ) from exc
                trigger_labels[b, :n] = torch.tensor(record["triggers"], dtype=torch.float32)

        encoded = self.tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_utterance_tokens,
            return_tensors="pt",
        )
        seq_len = encoded["input_ids"].shape[-1]
        result: dict[str, Any] = {
            "input_ids": encoded["input_ids"].view(batch_size, max_u, seq_len),
            "attention_mask": encoded["attention_mask"].view(batch_size, max_u, seq_len),
            "utterance_mask": utterance_mask,
            "speaker_ids": speaker_ids,
            "records": batch,
        }
        if self.include_labels:
            result["emotion_labels"] = emotion_labels
            result["trigger_labels"] = trigger_labels
        return result
