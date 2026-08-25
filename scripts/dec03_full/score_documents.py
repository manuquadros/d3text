"""Per-class document-level scores for a checkpoint, on any split.

Comparing two pooling settings needs the number the training loss hides:
whether each class head ever predicts its class. `evaluate` reports this for
the test split only, inside a larger run; this scores one named split and
writes the per-class table to JSON, so two pooling arms can be compared
directly.
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from d3text import checkpoint, data, factory, runtime
from d3text.models.config import encodings, load_model_config

logger = logging.getLogger(__name__)


def command_line_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="score_documents")
    parser.add_argument("config")
    parser.add_argument("model_state_dict")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    runtime.configure()
    args = command_line_args()
    config = load_model_config(args.config)

    saved = checkpoint.load(args.model_state_dict)
    if saved.vocabulary is None:
        raise SystemExit("checkpoint records no vocabulary")

    dataset = data.brenda_dataset(
        encodings=encodings[config.base_model],
        vocabulary=saved.vocabulary,
        split_names=(args.split,),
    )
    loader = data.get_batch_loader(
        dataset=dataset.data[args.split],
        batch_size=1,
        max_chunks=config.batch_max_chunks,
    )

    model = factory.build_model(config, dataset)
    model.register_load_state_dict_pre_hook(factory.fix_keys_hook)
    model.load_state_dict(saved.state_dict)
    model.to(model.device)
    model.eval()

    logits, targets = [], []
    with torch.no_grad():
        for batch in loader:
            out = model.get_batch_logits(batch)
            cls_logits = out[1] if isinstance(out, tuple) else out
            truth = model.ground_truth(batch)
            cls_true = truth[1] if isinstance(truth, tuple) else truth
            logits.append(model.drop_oos(cls_logits).detach().float().cpu())
            targets.append(cls_true.detach().to(torch.int64).cpu())

    probs = torch.sigmoid(torch.cat(logits, dim=0)).numpy()
    true = torch.cat(targets, dim=0).numpy().astype(int)
    pred = (probs >= args.threshold).astype(int)

    report = {}
    for column, name in enumerate(model.known_classes):
        y, p, s = true[:, column], pred[:, column], probs[:, column]
        tp = int((y & p).sum())
        precision = tp / max(int(p.sum()), 1)
        recall = tp / max(int(y.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        report[name] = {
            "support": int(y.sum()),
            "predicted": int(p.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": float(average_precision_score(y, s)) if y.any() else None,
            "mean_prob_positive": float(s[y == 1].mean()) if y.any() else None,
            "mean_prob_negative": float(s[y == 0].mean())
            if (y == 0).any()
            else None,
        }

    macro_f1 = float(np.mean([c["f1"] for c in report.values()]))
    result = {
        "config": args.config,
        "checkpoint": args.model_state_dict,
        "split": args.split,
        "documents": int(true.shape[0]),
        "threshold": args.threshold,
        "pooling": config.entity_logits_pooling,
        "per_class": report,
        "macro_f1": macro_f1,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    logger.info("%s", json.dumps(result["per_class"], indent=2))
    logger.info("macro-F1 %.4f over %d documents", macro_f1, true.shape[0])


if __name__ == "__main__":
    main()
