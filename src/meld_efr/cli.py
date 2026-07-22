
from __future__ import annotations

import argparse
import json
from pathlib import Path



def _add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="Allena il modello e salva il best checkpoint.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path")
    parser.add_argument("--test-path")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--max-utterance-tokens", type=int, default=64)
    parser.add_argument("--max-speakers", type=int, default=16)
    parser.add_argument("--speaker-dim", type=int, default=32)
    parser.add_argument("--context-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--trigger-loss-weight", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")


def _add_predict_parser(subparsers) -> None:
    parser = subparsers.add_parser("predict", help="Esegue inference su uno o più dialoghi JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", dest="output_path")
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meld-efr")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_predict_parser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        from .training import TrainConfig, train

        config = TrainConfig(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_utterance_tokens=args.max_utterance_tokens,
            max_speakers=args.max_speakers,
            speaker_dim=args.speaker_dim,
            context_hidden=args.context_hidden,
            dropout=args.dropout,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            patience=args.patience,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            trigger_loss_weight=args.trigger_loss_weight,
            label_smoothing=args.label_smoothing,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
        )
        train(config)
    elif args.command == "predict":
        from .inference import predict_records

        results = predict_records(
            checkpoint_path=args.checkpoint,
            input_path=args.input_path,
            output_path=args.output_path,
            device_name=args.device,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if args.output_path:
            print(f"\nPredizioni salvate in: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
