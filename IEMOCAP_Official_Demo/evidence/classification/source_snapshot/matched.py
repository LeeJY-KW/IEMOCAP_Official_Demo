"""Fixed, paired EmotionHeart comparisons; the session trainer remains auxiliary."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import train as shared
from .data import load_manifest

ARMS = {
    "eh_audio": {"family": "emotionheart", "input": "A", "missing": False},
    "graph_audio": {"family": "graph", "input": "A", "missing": False},
    "eh_full": {"family": "emotionheart", "input": "ATV", "missing": False},
    "graph_full": {"family": "graph", "input": "ATV", "missing": False},
    "eh_missing": {"family": "emotionheart", "input": "ATV", "missing": True},
    "graph_missing": {"family": "graph", "input": "ATV", "missing": True},
}
PAIRS = {"audio": ("eh_audio", "graph_audio"), "complete": ("eh_full", "graph_full"),
         "missing": ("eh_missing", "graph_missing")}
TEST_SCENARIOS = list(shared.SUBSETS) + [f"random_{p}" for p in (.1, .3, .5, .7)] + [f"block_{n}" for n in (2, 4, 8)]


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    local = Path(__file__).parent
    files = [local / name for name in ("matched.py", "train.py", "model.py", "data.py", "emotionheart_baseline.py")]
    files += sorted((root / "models" / "emotionheart").glob("*.py"))
    files += [root / "graphdata" / name for name in ("iemocap_graphDatasetLoader.py", "algos.py")]
    return {str(path.relative_to(root)): file_hash(path) for path in files}


def build_protocol(cache):
    manifest = load_manifest(cache)
    dialogues = sorted(manifest["dialogues"], key=lambda item: item["id"])
    ids = [item["id"] for item in dialogues]
    if len(ids) != 151 or len(set(ids)) != 151:
        raise ValueError("Matched protocol requires exactly 151 distinct dialogues")
    population = [[d["id"], [[u["id"], u["label"]] for u in d["utterances"] if u["label"] != -100]] for d in dialogues]
    if any(not targets for _, targets in population):
        raise ValueError("Empty dialogue after six-class filtering")
    if manifest["labels"] != ["ang", "hap", "exc", "sad", "fru", "neu"]:
        raise ValueError("Matched protocol requires the declared six-label order")
    repeats = []
    for seed in range(42, 52):
        shuffled = np.random.default_rng(seed).permutation(ids).tolist()
        repeats.append({"seed": seed, "train": shuffled[:105], "dev": shuffled[105:120], "test": shuffled[120:]})
    environment = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for name in ("fairseq", "transformers"):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = "unavailable"
    return {"schema_version": 1, "protocol_id": "matched-dialogue-v1",
            "manifest_hash": shared.digest(manifest), "feature_hash": shared.cache_fingerprint(cache, manifest),
            "cache_files": {identifier: file_hash(Path(cache) / "dialogues" / f"{identifier}.npz") for identifier in ids},
            "population_hash": shared.digest(population), "population": population,
            "n_targets": sum(len(targets) for _, targets in population), "labels": manifest["labels"],
            "context_policy": "six_class_only_reindexed_turns", "repeats": repeats, "arms": ARMS,
            "budget": {"epochs": 50, "batch_dialogues": 12, "lr": 1e-4, "weight_decay": 1e-5,
                       "optimizer": "AdamW", "schedule": "constant", "clipping": None, "early_stopping": False},
            "graph": {"hidden": 256, "layers": 2, "dropout": .2, "paths": [1, 2, 3]},
            "emotionheart": {"hidden": 384, "heads": 6, "layers": 2, "temperature": .7,
                             "dropout": .3, "objective": "0.5CE+0.3GCL", "padding": "train_max"},
            "selection": "development weighted_f1: A / ATV / mean seven subsets for missing arms",
            "missing_curriculum": {"full": .25, "partial": .5, "block": .25, "block_lengths": [2, 5]},
            "test_scenarios": TEST_SCENARIOS, "mask_seed": 2718,
            "sources": source_hashes(), "environment": environment,
            "limitations": ["Reconstructed dialogue-random splits; exact published IDs and split unit unavailable.",
                            "Raw corpus has 7380 six-class targets versus paper7433; no invented examples.",
                            "Ten repeats overlap and are not independent samples.",
                            "Prior pilot informed method development; published scores are historical references only."]}


def protocol(cache, output, create=False):
    path = Path(output) / "protocol.json"
    current = build_protocol(cache)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != current:
            raise ValueError("Stale source/cache/environment/protocol; use a separate campaign directory")
        return saved
    if not create:
        raise ValueError("Prepare and freeze protocol.json before training")
    shared.write_json(path, current)
    return current


def six_class_only(dialogues):
    retained = []
    for item in dialogues:
        keep = item["labels"] != -100
        if not keep.any():
            raise ValueError(f"No six-class utterances in {item['dialogue_id']}")
        filtered = dict(item)
        for name in (*shared.FEATURES, "labels", "speaker", "order", "times", "ids"):
            filtered[name] = item[name][keep].copy()
        filtered["order"] = np.arange(int(keep.sum()), dtype=np.int64)
        retained.append(filtered)
    return retained


def dev_scenarios(arm):
    config = ARMS[arm]
    return list(shared.SUBSETS) if config["missing"] else [config["input"]]


def development_score(metrics, arm):
    return float(np.mean([metrics[name]["weighted_f1"] for name in dev_scenarios(arm)]))


def make_model(arm, dims, classes, pad_to, device):
    config = ARMS[arm]
    if config["family"] == "emotionheart":
        from .emotionheart_baseline import EmotionHeartBaseline
        return EmotionHeartBaseline(dims, num_classes=classes, modalities=config["input"], pad_to=pad_to).to(device)
    variant = "audio_graph" if config["input"] == "A" else "multi_graph"
    return shared.make_model(dict(shared.DEFAULT, variant=variant), dims, classes, device)


def epoch_plan(data, seed, epoch):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).permutation(len(data)).tolist()
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1]))
    masks, regimes = {}, {}
    for index in order:
        batch = data[index]
        mask, regime = shared.sample_training_mask(len(batch["ids"]), rng)
        masks[batch["dialogue_id"]] = mask
        regimes[regime] = regimes.get(regime, 0) + 1
    return order, masks, regimes


def group_objective(model, batches, masks, arm):
    config = ARMS[arm]

    def objective(observed):
        if config["family"] == "emotionheart":
            return model.loss_group(batches, masks=observed)
        losses = []
        for index, batch in enumerate(batches):
            mask = None if observed is None else observed[index]
            logits = model(batch, mask=mask)
            losses.append(F.cross_entropy(logits, batch["labels"], reduction="sum"))
        return torch.stack(losses).sum() / sum(len(batch["labels"]) for batch in batches)

    full = objective(None)
    return .5 * (full + objective(masks)) if config["missing"] else full


def run_identity(declaration, seed, arm):
    return shared.digest({"protocol": shared.digest(declaration), "seed": seed, "arm": arm})


def run_path(output, seed, arm):
    return Path(output) / f"seed{seed}" / arm


def check_lock(path, identity, epochs):
    lock = json.loads((path / "selection.json").read_text())
    if lock["identity"] != identity or lock["epochs_completed"] != epochs:
        raise ValueError(f"Incomplete/stale development lock: {path}")
    if file_hash(path / "best.pt") != lock["checkpoint_sha256"]:
        raise ValueError(f"Locked checkpoint changed: {path}")
    return lock


def save_history(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)


def train_run(cache, output, seed, arm, device="cuda"):
    declaration = protocol(cache, output)
    if arm not in declaration["arms"]:
        raise ValueError(f"Unknown arm: {arm}")
    split = next((item for item in declaration["repeats"] if item["seed"] == seed), None)
    if split is None:
        raise ValueError("Training seed must be one of the ten declared repeats")
    budget = declaration["budget"]
    path, identity = run_path(output, seed, arm), run_identity(declaration, seed, arm)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "selection.json").exists():
        return check_lock(path, identity, budget["epochs"])
    shared.write_json(path / "config.json", {"identity": identity, "protocol_hash": shared.digest(declaration),
                                           "seed": seed, "arm": arm, "split": split, "budget": budget})
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    shared.set_seed(seed)
    rng = np.random.default_rng(seed)
    manifest = load_manifest(cache)
    raw_train = six_class_only(shared.load_dialogues(cache, split["train"], manifest))
    raw_dev = six_class_only(shared.load_dialogues(cache, split["dev"], manifest))
    stats = shared.fit_normalizer(raw_train)
    dims = {name: raw_train[0][name].shape[1] for name in shared.FEATURES}
    pad_to = max(len(item["labels"]) for item in raw_train)
    train_data = shared.normalize_dialogues(raw_train, stats, device)
    dev_data = shared.normalize_dialogues(raw_dev, stats, device)
    del raw_train, raw_dev
    model = make_model(arm, dims, len(declaration["labels"]), pad_to, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"])
    start, best, best_epoch, best_model, best_metrics = 1, -1., 0, None, None
    history = []
    restart = path / "last.pt"
    if restart.exists():
        state = torch.load(restart, map_location=device, weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Stale restart checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        shared.restore_random_state(state["rng"], rng)
        start, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
        best_model = {name: value.cpu() for name, value in state["best_model"].items()}
        best_metrics = state["best_metrics"]
        history = state["history"]
        del state
    history_path = path / "history.jsonl"
    save_history(history_path, history)  # Restart owns history, including a lost/partial JSONL tail.
    for epoch in range(start, budget["epochs"] + 1):
        began = time.time()
        model.train()
        order, masks, regimes = epoch_plan(train_data, seed, epoch)
        order_ids = [train_data[index]["dialogue_id"] for index in order]
        if ARMS[arm]["missing"]:
            np.savez_compressed(path / f"train_masks_epoch{epoch:02d}.npz", **masks)
        total_loss, total_count = 0., 0
        for offset in range(0, len(order), budget["batch_dialogues"]):
            batches = [train_data[index] for index in order[offset:offset + budget["batch_dialogues"]]]
            observed = [torch.as_tensor(masks[b["dialogue_id"]], device=device) for b in batches]
            optimizer.zero_grad(set_to_none=True)
            loss = group_objective(model, batches, observed, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {arm} loss at epoch {epoch}")
            loss.backward()
            finite = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            if not finite or not torch.stack(finite).all():
                raise FloatingPointError(f"Nonfinite/missing {arm} gradient at epoch {epoch}")
            optimizer.step()
            count = sum(len(batch["labels"]) for batch in batches)
            total_loss += float(loss.detach()) * count
            total_count += count
        metrics = shared.evaluate_model(model, dev_data, scenarios=dev_scenarios(arm))
        score = development_score(metrics, arm)
        if score > best:
            best, best_epoch, best_model, best_metrics = score, epoch, shared.snapshot(model), metrics
        row = {"epoch": epoch, "loss": total_loss / total_count, "dev_weighted_f1": score,
               "best": best, "best_epoch": best_epoch, "seconds": time.time() - began,
               "order_hash": shared.digest(order_ids), "mask_hash": shared.digest({k: v.tolist() for k, v in masks.items()}),
               "sampled_regimes": regimes, "observed_fraction": np.concatenate(list(masks.values())).mean(0).tolist()
               if ARMS[arm]["missing"] else ([1., 0., 0.] if ARMS[arm]["input"] == "A" else [1., 1., 1.]), "dev": metrics}
        history.append(row)
        shared.save_torch(restart, {"identity": identity, "model": shared.snapshot(model),
                                   "optimizer": optimizer.state_dict(), "rng": shared.random_state(rng),
                                   "epoch": epoch, "best": best, "best_epoch": best_epoch,
                                   "best_model": best_model, "best_metrics": best_metrics, "history": history})
        save_history(history_path, history)
        print(f"matched seed={seed} {arm} epoch={epoch}/50 loss={row['loss']:.4f} "
              f"dev_wF1={score:.4f} best={best:.4f}@{best_epoch} seconds={row['seconds']:.1f}", flush=True)
    model.load_state_dict(best_model)
    checkpoint = {"identity": identity, "model": shared.snapshot(model), "normalizer": stats, "dims": dims,
                  "classes": len(declaration["labels"]), "pad_to": pad_to, "arm": arm, "seed": seed,
                  "epoch": best_epoch, "dev_score": best, "protocol_hash": shared.digest(declaration)}
    shared.save_torch(path / "best.pt", checkpoint)
    shared.write_json(path / "dev.json", best_metrics)
    shared.evaluate_model(model, dev_data, scenarios=dev_scenarios(arm), output=path / "dev_predictions.npz")
    lock = {"identity": identity, "arm": arm, "seed": seed, "epochs_completed": budget["epochs"],
            "best_epoch": best_epoch, "dev_weighted_f1": best, "checkpoint_sha256": file_hash(path / "best.pt"),
            "normalizer_hash": shared.digest({name: {key: value.tolist() for key, value in fields.items()}
                                                for name, fields in stats.items()}),
            "test_evaluated_during_training": False}
    shared.write_json(path / "selection.json", lock)
    return lock


def require_all_locks(output, declaration):
    locks = []
    for split in declaration["repeats"]:
        for arm in declaration["arms"]:
            path = run_path(output, split["seed"], arm)
            if not (path / "selection.json").exists():
                raise ValueError(f"No test access: all 60 development runs must be locked; missing {path}")
            locks.append(check_lock(path, run_identity(declaration, split["seed"], arm), declaration["budget"]["epochs"]))
    return locks


def descriptive(values):
    values = np.asarray(values, dtype=float)
    return {"mean": values.mean(axis=0).tolist(), "sample_sd": values.std(axis=0, ddof=1).tolist()}


def audit_pairs(output, declaration, predictions=False):
    for split in declaration["repeats"]:
        for pair, (baseline, proposed) in PAIRS.items():
            left, right = [run_path(output, split["seed"], arm) for arm in (baseline, proposed)]
            norms = [json.loads((path / "selection.json").read_text())["normalizer_hash"] for path in (left, right)]
            if norms[0] != norms[1]:
                raise ValueError("Paired train-only normalizer mismatch")
            histories = [[json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
                         for path in (left, right)]
            if any([row["epoch"] for row in history] != list(range(1, 51)) for history in histories):
                raise ValueError("Paired comparison requires complete 50-epoch histories")
            for first, second in zip(*histories):
                if first["order_hash"] != second["order_hash"] or (pair == "missing" and first["mask_hash"] != second["mask_hash"]):
                    raise ValueError("Paired training order/mask mismatch")
            if predictions:
                with np.load(left / "test_predictions.npz") as first, np.load(right / "test_predictions.npz") as second:
                    for name in first.files:
                        if name.endswith(("_ids", "_labels", "_masks")) and not np.array_equal(first[name], second[name]):
                            raise ValueError(f"Paired evaluation population/mask mismatch: {name}")


def aggregate(rows):
    indexed = {(row["seed"], row["arm"]): row["metrics"] for row in rows}
    # Compute each repeat's seven-subset mean first; SD must be across repeat means.
    for metrics in indexed.values():
        if all(name in metrics for name in shared.SUBSETS):
            metrics["seven_subset_mean"] = {metric: np.mean([metrics[name][metric] for name in shared.SUBSETS], axis=0).tolist()
                                             for metric in ("macro_f1", "weighted_f1", "uar", "accuracy", "per_class_f1", "confusion")}
    result = {"arms": {}, "paired_differences_pp": {},
              "interpretation": "Mean/sample SD across ten overlapping paired repeats; no independent-sample significance claim."}
    for arm in ARMS:
        selected = [indexed[seed, arm] for seed in range(42, 52)]
        result["arms"][arm] = {}
        for scenario in selected[0]:
            result["arms"][arm][scenario] = {name: descriptive([value[scenario][name] for value in selected])
                                             for name in ("macro_f1", "weighted_f1", "uar", "accuracy", "per_class_f1", "confusion")}
    for name, (baseline, proposed) in PAIRS.items():
        result["paired_differences_pp"][name] = {}
        for scenario in indexed[42, baseline]:
            result["paired_differences_pp"][name][scenario] = {}
            for metric in ("macro_f1", "weighted_f1", "uar", "accuracy"):
                differences = [100 * (indexed[seed, proposed][scenario][metric] - indexed[seed, baseline][scenario][metric]) for seed in range(42, 52)]
                result["paired_differences_pp"][name][scenario][metric] = dict(descriptive(differences), repeats=differences)
    return result


def evaluate_all(cache, output, device="cuda"):
    declaration = protocol(cache, output)
    locks = require_all_locks(output, declaration)  # Must precede any test cache loading.
    audit_pairs(output, declaration)
    shared.write_json(Path(output) / "evaluation_gate.json", {"protocol_hash": shared.digest(declaration), "locks": locks})
    manifest = load_manifest(cache)
    rows = []
    for split in declaration["repeats"]:
        raw_test = six_class_only(shared.load_dialogues(cache, split["test"], manifest))
        for arm in declaration["arms"]:
            path, identity = run_path(output, split["seed"], arm), run_identity(declaration, split["seed"], arm)
            result_path = path / "test.json"
            if result_path.exists():
                row = json.loads(result_path.read_text())
                if row["identity"] != identity:
                    raise ValueError("Stale test result")
            else:
                saved = torch.load(path / "best.pt", map_location=device, weights_only=False)
                if saved["identity"] != identity:
                    raise ValueError("Stale evaluation checkpoint")
                model = make_model(arm, saved["dims"], saved["classes"], saved["pad_to"], device)
                model.load_state_dict(saved["model"])
                data = shared.normalize_dialogues(raw_test, saved["normalizer"], device)
                scenarios = ["A"] if ARMS[arm]["input"] == "A" else declaration["test_scenarios"]
                metrics = shared.evaluate_model(model, data, scenarios=scenarios, output=path / "test_predictions.npz")
                row = {"identity": identity, "arm": arm, "seed": split["seed"], "metrics": metrics}
                shared.write_json(result_path, row)
                del model, data, saved
            rows.append(row)
            shared.write_json(Path(output) / "test_results.json", rows)
    audit_pairs(output, declaration, predictions=True)
    summary = aggregate(rows)
    shared.write_json(Path(output) / "test_summary.json", summary)
    return summary


def campaign(cache, output, device="cuda", train_only=False):
    declaration = protocol(cache, output, create=True)
    for split in declaration["repeats"]:
        for arm in declaration["arms"]:
            train_run(cache, output, split["seed"], arm, device)
    return require_all_locks(output, declaration) if train_only else evaluate_all(cache, output, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "campaign", "evaluate"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=range(42, 52), default=42)
    parser.add_argument("--arm", choices=tuple(ARMS), default="graph_audio")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        frozen = protocol(args.cache, args.output, create=True)
        print(json.dumps({"protocol_hash": shared.digest(frozen), "targets": frozen["n_targets"], "runs": 60}))
    elif args.command == "train":
        train_run(args.cache, args.output, args.seed, args.arm, args.device)
    elif args.command == "evaluate":
        evaluate_all(args.cache, args.output, args.device)
    else:
        campaign(args.cache, args.output, args.device, args.train_only)


if __name__ == "__main__":
    main()
