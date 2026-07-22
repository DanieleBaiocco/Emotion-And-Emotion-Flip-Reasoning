
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import (
    DialogueCollator,
    DialogueDataset,
    build_emotion_mapping,
    flatten_field,
    load_records,
    split_train_val_test,
)
from .model import DialogueMultiTaskModel
from .utils import get_device, set_seed, write_json


@dataclass
class TrainConfig:
    train_path: str
    output_dir: str
    val_path: str | None = None
    test_path: str | None = None
    model_name: str = "bert-base-uncased"
    max_utterance_tokens: int = 64
    max_speakers: int = 16
    speaker_dim: int = 32
    context_hidden: int = 256
    dropout: float = 0.25
    batch_size: int = 4
    grad_accum_steps: int = 2
    epochs: int = 8
    patience: int = 3
    encoder_lr: float = 2e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    trigger_loss_weight: float = 0.7
    label_smoothing: float = 0.05
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _compute_class_weights(
    records: list[dict[str, Any]], emotion_to_id: dict[str, int]
) -> tuple[torch.Tensor, float]:
    emotion_ids = [emotion_to_id[e] for e in flatten_field(records, "emotions")]
    counts = np.bincount(emotion_ids, minlength=len(emotion_to_id)).astype(np.float64)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.35, 4.0)

    triggers = np.asarray(flatten_field(records, "triggers"), dtype=np.float32)
    positives = float((triggers > 0.5).sum())
    negatives = float((triggers <= 0.5).sum())
    pos_weight = float(np.clip(negatives / max(positives, 1.0), 1.0, 8.0))
    return torch.tensor(weights, dtype=torch.float32), pos_weight


def _compute_loss(
    outputs: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, Any],
    emotion_weights: torch.Tensor,
    trigger_pos_weight: float,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    emotion_logits, trigger_logits = outputs
    mask = batch["utterance_mask"]
    emotion_loss = F.cross_entropy(
        emotion_logits[mask],
        batch["emotion_labels"][mask],
        weight=emotion_weights,
        label_smoothing=config.label_smoothing,
    )
    pos_weight = torch.tensor(trigger_pos_weight, device=trigger_logits.device)
    trigger_loss = F.binary_cross_entropy_with_logits(
        trigger_logits[mask], batch["trigger_labels"][mask], pos_weight=pos_weight
    )
    total = emotion_loss + config.trigger_loss_weight * trigger_loss
    return total, emotion_loss.detach(), trigger_loss.detach()


def _build_optimizer_scheduler(
    model: DialogueMultiTaskModel, config: TrainConfig, num_batches: int
):
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(num_batches / config.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


def _metrics(
    emotion_true: list[int],
    emotion_pred: list[int],
    trigger_true: list[int],
    trigger_prob: list[float],
    threshold: float,
) -> dict[str, float]:
    trigger_pred = (np.asarray(trigger_prob) >= threshold).astype(int)
    return {
        "emotion_accuracy": float(accuracy_score(emotion_true, emotion_pred)),
        "emotion_macro_f1": float(
            f1_score(emotion_true, emotion_pred, average="macro", zero_division=0)
        ),
        "emotion_weighted_f1": float(
            f1_score(emotion_true, emotion_pred, average="weighted", zero_division=0)
        ),
        "trigger_accuracy": float(accuracy_score(trigger_true, trigger_pred)),
        "trigger_f1_positive": float(
            f1_score(trigger_true, trigger_pred, average="binary", zero_division=0)
        ),
        "trigger_macro_f1": float(
            f1_score(trigger_true, trigger_pred, average="macro", zero_division=0)
        ),
    }


@torch.no_grad()
def _evaluate(
    model: DialogueMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    emotion_weights: torch.Tensor,
    trigger_pos_weight: float,
    config: TrainConfig,
    threshold: float = 0.5,
):
    model.eval()
    losses: list[float] = []
    emotion_true: list[int] = []
    emotion_pred: list[int] = []
    trigger_true: list[int] = []
    trigger_prob: list[float] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["utterance_mask"],
            batch["speaker_ids"],
        )
        loss, _, _ = _compute_loss(
            outputs, batch, emotion_weights, trigger_pos_weight, config
        )
        losses.append(float(loss.item()))
        emotion_logits, trigger_logits = outputs
        mask = batch["utterance_mask"]
        emotion_true.extend(batch["emotion_labels"][mask].cpu().tolist())
        emotion_pred.extend(emotion_logits[mask].argmax(-1).cpu().tolist())
        trigger_true.extend(batch["trigger_labels"][mask].int().cpu().tolist())
        trigger_prob.extend(torch.sigmoid(trigger_logits[mask]).cpu().tolist())

    metrics = _metrics(
        emotion_true, emotion_pred, trigger_true, trigger_prob, threshold=threshold
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    arrays = {
        "emotion_true": emotion_true,
        "emotion_pred": emotion_pred,
        "trigger_true": trigger_true,
        "trigger_prob": trigger_prob,
    }
    return metrics, arrays


def _tune_threshold(trigger_true: list[int], trigger_prob: list[float]) -> tuple[float, float]:
    thresholds = np.arange(0.10, 0.91, 0.05)
    scores = [
        f1_score(
            trigger_true,
            np.asarray(trigger_prob) >= threshold,
            average="binary",
            zero_division=0,
        )
        for threshold in thresholds
    ]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def _save_checkpoint(
    path: Path,
    model: DialogueMultiTaskModel,
    emotion_to_id: dict[str, int],
    config: TrainConfig,
    trigger_threshold: float,
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "architecture": model.architecture_config(),
        "emotion_to_id": emotion_to_id,
        "trigger_threshold": float(trigger_threshold),
        "training_config": asdict(config),
        "metrics": metrics or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = get_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"

    train_records = load_records(config.train_path, require_labels=True)
    if bool(config.val_path) != bool(config.test_path):
        raise ValueError("Passa sia --val-path sia --test-path, oppure nessuno dei due.")
    if config.val_path and config.test_path:
        val_records = load_records(config.val_path, require_labels=True)
        test_records = load_records(config.test_path, require_labels=True)
    else:
        train_records, val_records, test_records = split_train_val_test(
            train_records, config.seed
        )

    emotion_to_id = build_emotion_mapping(train_records)
    id_to_emotion = {idx: label for label, idx in emotion_to_id.items()}
    print(f"Device: {device}")
    print(f"Dialoghi: train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    print("Emotion classes:", list(emotion_to_id))
    print("Train emotion distribution:", dict(Counter(flatten_field(train_records, "emotions"))))

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = DialogueCollator(
        tokenizer,
        emotion_to_id,
        max_utterance_tokens=config.max_utterance_tokens,
        max_speakers=config.max_speakers,
        include_labels=True,
    )
    loader_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(DialogueDataset(train_records), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(DialogueDataset(val_records), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(DialogueDataset(test_records), shuffle=False, **loader_kwargs)

    model = DialogueMultiTaskModel(
        model_name=config.model_name,
        num_emotions=len(emotion_to_id),
        max_speakers=config.max_speakers,
        speaker_dim=config.speaker_dim,
        context_hidden=config.context_hidden,
        dropout=config.dropout,
    ).to(device)
    # Salviamo tokenizer + config dell'encoder nel progetto, così l'inference del
    # checkpoint non deve riscaricare bert-base-uncased da Hugging Face.
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.encoder.config.save_pretrained(output_dir / "encoder_config")
    emotion_weights, trigger_pos_weight = _compute_class_weights(
        train_records, emotion_to_id
    )
    emotion_weights = emotion_weights.to(device)
    optimizer, scheduler = _build_optimizer_scheduler(model, config, len(train_loader))

    # API AMP compatibile con versioni torch recenti e meno recenti.
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "emotion_loss": 0.0, "trigger_loss": 0.0}

        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch(batch, device)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                amp_context = torch.amp.autocast("cuda", enabled=use_amp)
            else:
                amp_context = torch.cuda.amp.autocast(enabled=use_amp)

            with amp_context:
                outputs = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["utterance_mask"],
                    batch["speaker_ids"],
                )
                loss, emotion_loss, trigger_loss = _compute_loss(
                    outputs, batch, emotion_weights, trigger_pos_weight, config
                )
                backward_loss = loss / config.grad_accum_steps

            scaler.scale(backward_loss).backward()
            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running["loss"] += float(loss.item())
            running["emotion_loss"] += float(emotion_loss.item())
            running["trigger_loss"] += float(trigger_loss.item())

        val_metrics, _ = _evaluate(
            model,
            val_loader,
            device,
            emotion_weights,
            trigger_pos_weight,
            config,
            threshold=0.5,
        )
        score = val_metrics["emotion_macro_f1"] + val_metrics["trigger_f1_positive"]
        row = {
            "epoch": epoch,
            "train_loss": running["loss"] / max(1, len(train_loader)),
            "train_emotion_loss": running["emotion_loss"] / max(1, len(train_loader)),
            "train_trigger_loss": running["trigger_loss"] / max(1, len(train_loader)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        print(
            f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"emotion_macro_f1={val_metrics['emotion_macro_f1']:.4f} | "
            f"trigger_f1+={val_metrics['trigger_f1_positive']:.4f}"
        )

        if score > best_score + 1e-4:
            best_score = score
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                emotion_to_id,
                config,
                trigger_threshold=0.5,
                metrics={"best_validation_score": float(best_score), "epoch": epoch},
            )
            print(f"  -> salvato nuovo best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping alla epoch {epoch}.")
                break

    if not checkpoint_path.exists():
        raise RuntimeError("Training terminato senza produrre un checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_arrays = _evaluate(
        model,
        val_loader,
        device,
        emotion_weights,
        trigger_pos_weight,
        config,
        threshold=0.5,
    )
    best_threshold, tuned_val_f1 = _tune_threshold(
        val_arrays["trigger_true"], val_arrays["trigger_prob"]
    )
    test_metrics, test_arrays = _evaluate(
        model,
        test_loader,
        device,
        emotion_weights,
        trigger_pos_weight,
        config,
        threshold=best_threshold,
    )

    majority_emotion = Counter(flatten_field(train_records, "emotions")).most_common(1)[0][0]
    majority_id = emotion_to_id[majority_emotion]
    test_emotions = [emotion_to_id[e] for e in flatten_field(test_records, "emotions")]
    baseline_emotion_pred = [majority_id] * len(test_emotions)
    baseline_trigger_true = [int(x > 0.5) for x in flatten_field(test_records, "triggers")]
    baseline_metrics = _metrics(
        test_emotions,
        baseline_emotion_pred,
        baseline_trigger_true,
        [0.0] * len(baseline_trigger_true),
        threshold=0.5,
    )

    predicted_distribution = Counter(test_arrays["emotion_pred"])
    diagnostics = {
        "majority_emotion": majority_emotion,
        "predicted_emotion_distribution": {
            id_to_emotion[i]: int(predicted_distribution.get(i, 0))
            for i in range(len(id_to_emotion))
        },
        "emotion_macro_f1_beats_majority": bool(
            test_metrics["emotion_macro_f1"] > baseline_metrics["emotion_macro_f1"]
        ),
        "num_predicted_emotion_classes": int(len(predicted_distribution)),
    }
    final_metrics = {
        "validation_at_0_5": val_metrics,
        "tuned_trigger_threshold": best_threshold,
        "validation_trigger_f1_at_tuned_threshold": tuned_val_f1,
        "test": test_metrics,
        "baseline": baseline_metrics,
        "diagnostics": diagnostics,
    }

    _save_checkpoint(
        checkpoint_path,
        model,
        emotion_to_id,
        config,
        trigger_threshold=best_threshold,
        metrics=final_metrics,
    )
    write_json(output_dir / "metrics.json", final_metrics)
    write_json(output_dir / "training_config.json", asdict(config))
    write_json(output_dir / "label_mapping.json", emotion_to_id)

    print("\nTraining completato.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trigger threshold: {best_threshold:.2f}")
    print(f"Test emotion macro-F1: {test_metrics['emotion_macro_f1']:.4f}")
    print(f"Majority emotion macro-F1: {baseline_metrics['emotion_macro_f1']:.4f}")
    print(f"Test trigger F1+: {test_metrics['trigger_f1_positive']:.4f}")
    if not diagnostics["emotion_macro_f1_beats_majority"]:
        print("WARNING: il modello non supera la majority baseline sulle emozioni.")
    if diagnostics["num_predicted_emotion_classes"] <= 2:
        print("WARNING: possibile collasso: il modello usa <= 2 classi emotion sul test.")
    return final_metrics
