"""Session-held-out training and bounded, development-only model selection.

Run from the repository root with ``python -m emotionheart.inductive.train``.
Only ``evaluate --split test`` and a locked final campaign read held-out caches.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

FEATURES = ("audio", "prosody", "text", "visual")
SUBSETS = {"A": (1, 0, 0), "T": (0, 1, 0), "V": (0, 0, 1),
           "AT": (1, 1, 0), "AV": (1, 0, 1), "TV": (0, 1, 1), "ATV": (1, 1, 1)}
DEFAULT = dict(variant="audio_graph", hidden=256, layers=2, dropout=.2,
               lr=1e-4, weight_decay=1e-4, max_epochs=100, patience=10,
               accumulation=8, seed=17, fold=1, class_weight=False,
               consistency=0., consistency_warmup=5, confidence=.7,
               paths=[1, 2, 3], relative=True, intermediate=True,
               path_attention=True, window=8, max_neighbors=8)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def session_number(value):
    return int(str(value).lower().replace("session", ""))


def split_dialogues(manifest, fold):
    if fold not in range(1, 6):
        raise ValueError("fold must be a session number 1..5")
    dev = fold % 5 + 1
    result = {"train": [], "dev": [], "test": []}
    for dialogue in manifest["dialogues"]:
        session = session_number(dialogue["session"])
        if session not in range(1, 6):
            raise ValueError(f"Invalid IEMOCAP session: {session}")
        role = "test" if session == fold else "dev" if session == dev else "train"
        result[role].append(dialogue["id"])
    flattened = sum(result.values(), [])
    if len(flattened) != len(set(flattened)) or any(not ids for ids in result.values()):
        raise ValueError("Duplicate dialogues or empty session split")
    return result


def load_dialogues(cache, ids, manifest):
    fingerprint = cache_fingerprint(cache, manifest)
    indexed = {d["id"]: d for d in manifest["dialogues"]}
    dialogues = []
    for identifier in ids:
        with np.load(Path(cache) / "dialogues" / f"{identifier}.npz", allow_pickle=False) as saved:
            data = {key: saved[key].copy() for key in saved.files}
        if str(data["fingerprint"]) != fingerprint:
            raise ValueError(f"Stale extractor provenance in {identifier}")
        expected = [u["id"] for u in indexed[identifier]["utterances"]]
        if data["ids"].tolist() != expected:
            raise ValueError(f"Cache/manifest utterance mismatch in {identifier}")
        if data["labels"].tolist() != [u["label"] for u in indexed[identifier]["utterances"]]:
            raise ValueError(f"Cache/manifest label mismatch in {identifier}")
        n = len(expected)
        for name in FEATURES:
            if data[name].ndim != 2 or data[name].shape[0] != n or not np.isfinite(data[name]).all():
                raise ValueError(f"Invalid {name} features in {identifier}")
        if (data["labels"].shape != (n,) or np.any((data["labels"] < 0) & (data["labels"] != -100))
                or np.any(data["labels"] >= len(manifest["labels"]))):
            raise ValueError(f"Invalid labels in {identifier}")
        data["dialogue_id"] = identifier
        dialogues.append(data)
    return dialogues


def cache_fingerprint(cache, manifest):
    from .data import extraction_fingerprint
    provenance = {name: json.loads((Path(cache) / f"extractor_{name}.json").read_text())
                  for name in FEATURES}
    return digest({name: extraction_fingerprint(manifest, value) for name, value in provenance.items()})


def fit_normalizer(train):
    """Fit only explicitly supplied train dialogues; never accept a full corpus."""
    stats = {}
    for name in FEATURES:
        values = np.concatenate([item[name] for item in train], axis=0).astype(np.float64)
        stats[name] = {"mean": values.mean(0).astype(np.float32),
                       "std": np.maximum(values.std(0), 1e-5).astype(np.float32)}
    return stats


def normalize_dialogues(dialogues, stats, device):
    output = []
    for item in dialogues:
        batch = {"dialogue_id": item["dialogue_id"], "ids": item["ids"].tolist()}
        for name in FEATURES:
            normalized = (item[name] - stats[name]["mean"]) / stats[name]["std"]
            batch[name] = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        for name in ("labels", "speaker", "order", "times"):
            batch[name] = torch.as_tensor(item[name], device=device)
        output.append(batch)
    return output


def classification_metrics(y, prediction, classes=6):
    y, prediction = np.asarray(y), np.asarray(prediction)
    valid = y != -100
    y, prediction = y[valid], prediction[valid]
    if not len(y):
        raise ValueError("No labeled targets to score")
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (y, prediction), 1)
    true_count, pred_count, correct = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(correct, true_count, out=np.zeros(classes, float), where=true_count > 0)
    f1 = np.divide(2 * correct, true_count + pred_count, out=np.zeros(classes, float),
                   where=(true_count + pred_count) > 0)
    return {"macro_f1": float(f1.mean()), "uar": float(recall.mean()),
            "weighted_f1": float(np.dot(f1, true_count) / len(y)),
            "accuracy": float(correct.sum() / len(y)), "n": int(len(y)),
            "confusion": cm.tolist(), "per_class_f1": f1.tolist()}


def is_audio(config):
    return config["variant"].startswith("audio_")


def selection_score(metrics, audio):
    names = ("A",) if audio else tuple(SUBSETS)
    return float(np.mean([metrics[name]["macro_f1"] for name in names]))


def sample_training_mask(n, rng):
    regime = rng.choice(("full", "partial", "block"), p=(.25, .5, .25))
    mask = np.ones((n, 3), dtype=bool)
    if regime == "partial":
        options = np.asarray(list(SUBSETS.values())[:6], dtype=bool)
        mask = options[rng.integers(0, 6, n)]
    elif regime == "block":
        length = min(n, int(rng.integers(2, 6)))
        start = int(rng.integers(0, n - length + 1))
        mask[start:start + length, int(rng.integers(3))] = False
    return mask, regime


def fixed_mask(n, scenario, dialogue_id, seed=2718):
    """Label-independent masks stable across trials, devices, and process order."""
    if scenario in SUBSETS:
        return np.tile(np.array(SUBSETS[scenario], dtype=bool), (n, 1))
    key = f"{seed}:{dialogue_id}:{scenario}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
    mask = np.ones((n, 3), dtype=bool)
    if scenario.startswith("random_"):
        probability = float(scenario.split("_")[1])
        if not 0 <= probability < 1:
            raise ValueError("Random missingness must be in [0,1)")
        mask = rng.random((n, 3)) >= probability
        while (~mask.any(1)).any():
            rows = ~mask.any(1)
            mask[rows] = rng.random((int(rows.sum()), 3)) >= probability
    elif scenario.startswith("block_"):
        length = min(n, int(scenario.split("_")[1]))
        start = int(rng.integers(0, n - length + 1))
        mask[start:start + length, int(rng.integers(3))] = False
    else:
        raise ValueError(f"Unknown missingness scenario {scenario}")
    return mask


@torch.no_grad()
def evaluate_model(model, data, audio=False, scenarios=None, output=None):
    model.eval()
    scenarios = scenarios or (("A",) if audio else tuple(SUBSETS))
    results, saved = {}, {}
    for scenario in scenarios:
        labels, predictions, probabilities, ids, masks, dialogue_ids = [], [], [], [], [], []
        for batch in data:
            mask = fixed_mask(len(batch["ids"]), scenario, batch["dialogue_id"])
            logits = model(batch, mask=torch.as_tensor(mask, device=batch["audio"].device))
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(logits.argmax(-1).cpu().tolist())
            probabilities.extend(logits.softmax(-1).cpu().tolist())
            ids.extend(batch["ids"])
            masks.extend(mask.tolist())
            dialogue_ids.extend([batch["dialogue_id"]] * len(mask))
        results[scenario] = classification_metrics(labels, predictions, logits.shape[-1])
        results[scenario]["observed_fraction"] = np.asarray(masks).mean(0).tolist()
        if output:
            saved[f"{scenario}_ids"] = np.asarray(ids)
            saved[f"{scenario}_dialogues"] = np.asarray(dialogue_ids)
            saved[f"{scenario}_labels"] = np.asarray(labels)
            saved[f"{scenario}_predictions"] = np.asarray(predictions)
            saved[f"{scenario}_probabilities"] = np.asarray(probabilities, dtype=np.float32)
            saved[f"{scenario}_masks"] = np.asarray(masks, dtype=bool)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **saved)
    return results


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_model(config, dims, classes, device):
    from .model import InductiveGraph
    keys = ("variant", "hidden", "layers", "dropout", "paths", "relative", "intermediate",
            "path_attention", "window", "max_neighbors")
    model = InductiveGraph(dims, num_classes=classes, **{k: config[k] for k in keys}).to(device)
    model.set_prosody_prior(torch.zeros(dims["prosody"], device=device),
                            torch.ones(dims["prosody"], device=device))
    return model


def supervised_loss(logits, labels, weights):
    valid = labels != -100
    return F.cross_entropy(logits[valid], labels[valid], weight=weights) if valid.any() else logits.sum() * 0


def training_loss(model, batch, config, epoch, rng, weights):
    labels = batch["labels"]
    if is_audio(config):
        return supervised_loss(model(batch), labels, weights), None, "audio"
    mask, regime = sample_training_mask(len(labels), rng)
    full_logits = model(batch)
    full_loss = supervised_loss(full_logits, labels, weights)
    if regime == "full":
        return full_loss, mask, regime
    masked_logits = model(batch, mask=torch.as_tensor(mask, device=labels.device))
    loss = .5 * (full_loss + supervised_loss(masked_logits, labels, weights))
    if config["consistency"] and epoch > config["consistency_warmup"]:
        teacher = full_logits.detach().softmax(-1)
        confident, predicted = teacher.max(-1)
        keep = (labels != -100) & (predicted == labels) & (confident >= config["confidence"])
        if keep.any():
            divergence = F.kl_div(masked_logits[keep].log_softmax(-1), teacher[keep], reduction="batchmean")
            loss = loss + config["consistency"] * divergence
    return loss, mask, regime


def random_state(rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "generator": copy.deepcopy(rng.bit_generator.state), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state, rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    rng.bit_generator.state = state["generator"]
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def train(cache, output, config=None, device="cuda"):
    from .data import load_manifest
    config = dict(DEFAULT, **(config or {}))
    if min(config["max_epochs"], config["patience"], config["accumulation"]) < 1:
        raise ValueError("Epochs, patience, and accumulation must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache)
    manifest_hash, feature_hash = digest(manifest), cache_fingerprint(cache, manifest)
    identity = {"config": config, "manifest_hash": manifest_hash, "feature_hash": feature_hash,
                "cache": str(cache.resolve())}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError(f"Refusing to overwrite a different run: {output}")
    write_json(config_path, identity)
    if (output / "done.json").exists():
        return json.loads((output / "done.json").read_text())
    set_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    splits = split_dialogues(manifest, config["fold"])
    write_json(output / "splits.json", splits)
    raw_train = load_dialogues(cache, splits["train"], manifest)
    raw_dev = load_dialogues(cache, splits["dev"], manifest)
    stats = fit_normalizer(raw_train)
    dims = {name: raw_train[0][name].shape[1] for name in FEATURES}
    classes = len(manifest["labels"])
    train_data = normalize_dialogues(raw_train, stats, device)
    dev_data = normalize_dialogues(raw_dev, stats, device)
    del raw_train, raw_dev
    model = make_model(config, dims, classes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weights = None
    if config["class_weight"]:
        labels = torch.cat([item["labels"] for item in train_data])
        counts = torch.bincount(labels[labels != -100], minlength=classes).float().clamp_min(1)
        weights = counts.rsqrt()
        weights = weights / weights.mean()
    best, best_epoch, start = -1., 0, 1
    last_path = output / "last.pt"
    if last_path.exists():
        resumed = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        best, best_epoch, start = resumed["best"], resumed["best_epoch"], resumed["epoch"] + 1
        restore_random_state(resumed["rng"], rng)
    started = time.time()
    for epoch in range(start, config["max_epochs"] + 1):
        if epoch - 1 - best_epoch >= config["patience"]:
            break
        epoch_start = time.time()
        model.train()
        order = rng.permutation(len(train_data)).tolist()
        total_loss, target_count = 0., 0
        regime_counts, observed, mask_count = {}, np.zeros(3, np.int64), 0
        for group_start in range(0, len(order), config["accumulation"]):
            group = order[group_start:group_start + config["accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                batch = train_data[index]
                loss, mask, regime = training_loss(model, batch, config, epoch, rng, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite train loss in {batch['dialogue_id']}")
                (loss / len(group)).backward()
                count = int((batch["labels"] != -100).sum())
                total_loss += float(loss.detach()) * count
                target_count += count
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                if mask is not None:
                    observed += mask.sum(0)
                    mask_count += len(mask)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        metrics = evaluate_model(model, dev_data, audio=is_audio(config))
        score = selection_score(metrics, is_audio(config))
        if score > best:
            best, best_epoch = score, epoch
            checkpoint = {"model": snapshot(model), "config": config, "dims": dims,
                          "classes": classes, "normalizer": stats, "manifest_hash": manifest_hash,
                          "feature_hash": feature_hash,
                          "splits": splits, "dev_score": best, "epoch": best_epoch}
            save_torch(output / "best.pt", checkpoint)
            write_json(output / "dev.json", metrics)
        row = {"epoch": epoch, "loss": total_loss / max(target_count, 1), "dev_score": score,
               "best_dev": best, "best_epoch": best_epoch, "seconds": time.time() - epoch_start,
               "regimes": regime_counts, "observed_fraction": (observed / mask_count).tolist() if mask_count else [1., 0., 0.],
               "dev": metrics}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        save_torch(last_path, {"model": snapshot(model), "optimizer": optimizer.state_dict(),
                              "epoch": epoch, "best": best, "best_epoch": best_epoch,
                              "rng": random_state(rng)})
        print(f"{output.name} epoch={epoch} loss={row['loss']:.4f} dev={score:.4f} "
              f"best={best:.4f}@{best_epoch} seconds={row['seconds']:.1f}", flush=True)
    selected = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    evaluate_model(model, dev_data, audio=is_audio(config), output=output / "dev_predictions.npz")
    done = {"config": config, "dev_score": best, "best_epoch": best_epoch,
            "elapsed_this_invocation_seconds": time.time() - started, "test_evaluated_during_training": False}
    write_json(output / "done.json", done)
    return done


def selection_identity(config):
    return {key: value for key, value in config.items() if key not in ("seed", "fold")}


def evaluate(cache, run, split="dev", device="cuda", lock=None):
    from .data import load_manifest
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    run, cache = Path(run), Path(cache)
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
    manifest = load_manifest(cache)
    if digest(manifest) != checkpoint["manifest_hash"]:
        raise ValueError("Manifest changed since training")
    if cache_fingerprint(cache, manifest) != checkpoint["feature_hash"]:
        raise ValueError("Feature extraction changed since training")
    if split == "test":
        if lock is None:
            raise ValueError("Test evaluation requires a frozen selection.json via --lock")
        selection = json.loads(Path(lock).read_text())
        if selection["manifest_hash"] != digest(manifest):
            raise ValueError("Selection lock does not match feature manifest")
        if selection["feature_hash"] != checkpoint["feature_hash"]:
            raise ValueError("Selection lock does not match feature extraction")
        if selection["fold"] != checkpoint["config"]["fold"]:
            raise ValueError("Selection lock belongs to another outer fold")
        candidates = [selection_identity(item["config"]) for item in selection["selected"].values()]
        if selection_identity(checkpoint["config"]) not in candidates:
            raise ValueError("Checkpoint configuration was not frozen in selection lock")
    raw = load_dialogues(cache, checkpoint["splits"][split], manifest)
    data = normalize_dialogues(raw, checkpoint["normalizer"], device)
    model = make_model(checkpoint["config"], checkpoint["dims"], checkpoint["classes"], device)
    model.load_state_dict(checkpoint["model"])
    audio = is_audio(checkpoint["config"])
    scenarios = ["A"] if audio else list(SUBSETS)
    if split == "test" and not audio:
        scenarios += [f"random_{p}" for p in (.1, .3, .5, .7)] + [f"block_{n}" for n in (2, 4, 8)]
    results = evaluate_model(model, data, audio, scenarios, run / f"{split}_predictions.npz")
    if not audio:
        results["summary"] = {"seven_subset_macro_f1": selection_score(results, False),
                              "audio_present_macro_f1": float(np.mean([results[n]["macro_f1"] for n in ("A", "AT", "AV", "ATV")]))}
    write_json(run / f"{split}.json", results)
    return results


def candidate_configs(track, base):
    """A bounded list fixed before test evaluation; each change remains auditable."""
    variant = "audio_graph" if track == "audio" else "multi_graph"
    baseline = "audio_mlp" if track == "audio" else "fusion"
    variants = [("baseline", {"variant": baseline}), ("graph", {"variant": variant}),
                ("baseline_lr3e4", {"variant": baseline, "lr": 3e-4}),
                ("baseline_dropout35", {"variant": baseline, "dropout": .35})]
    if track == "audio":
        variants += [("uniform", {"variant": "audio_uniform", "relative": False, "path_attention": False}),
                     ("homogeneous", {"variant": "audio_homo"})]
    else:
        variants += [("consistency", {"variant": variant, "consistency": .1}),
                     ("uniform_paths", {"variant": variant, "path_attention": False})]
    variants += [("lr3e4", {"variant": variant, "lr": 3e-4}),
                 ("lr5e5", {"variant": variant, "lr": 5e-5}),
                 ("dropout35", {"variant": variant, "dropout": .35}),
                 ("one_layer", {"variant": variant, "layers": 1}),
                 ("weighted", {"variant": variant, "class_weight": True}),
                 ("hidden128", {"variant": variant, "hidden": 128})]
    return [(name, dict(DEFAULT, **base, **changes)) for name, changes in variants]


def screen_fold(cache, output, fold, menu, manifest_hash, device):
    fold_path = output / f"fold{fold}"
    lock_path = fold_path / "selection.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["manifest_hash"] != manifest_hash or lock["candidate_menu_hash"] != digest(menu):
            raise ValueError("Locked selection does not match manifest/candidate menu")
        return lock, lock_path
    selected, all_trials = {}, []
    for track in ("audio", "multi"):
        best, misses = -1., 0
        for index, (name, template) in enumerate(menu["candidates"][track]):
            config = dict(template, fold=fold, seed=17)
            trial_path = fold_path / "screen" / f"{track}_{index:02d}_{name}"
            try:
                result = train(cache, trial_path, config, device)
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                trial = {"track": track, "name": name, "config": config,
                         "status": "failed", "error": str(error)}
                all_trials.append(trial)
                write_json(fold_path / "trials.json", all_trials)
                if name in ("baseline", "graph"):
                    raise
                misses += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if misses >= 4:
                    break
                continue
            trial = {"track": track, "name": name, "config": config, "status": "complete",
                     "dev_score": result["dev_score"], "path": str(trial_path.resolve())}
            all_trials.append(trial)
            if name.startswith("baseline"):
                previous = selected.get(f"{track}_baseline")
                if previous is None or trial["dev_score"] > previous["dev_score"]:
                    selected[f"{track}_baseline"] = trial
            # Controls are reported, but do not replace the requested graph family.
            eligible = config["variant"] in ("audio_graph", "multi_graph")
            if eligible and result["dev_score"] > best:
                selected[f"{track}_graph"] = trial
                best, misses = result["dev_score"], 0
            elif not name.startswith("baseline"):
                misses += 1
            write_json(fold_path / "trials.json", all_trials)
            print(f"screen fold={fold} {track}/{name} dev={result['dev_score']:.4f}, "
                  f"plateau={misses}/4", flush=True)
            if misses >= 4:
                break
    lock = {"schema_version": 2, "fold": fold, "manifest_hash": manifest_hash,
            "candidate_menu_hash": digest(menu), "feature_hash": menu["feature_hash"],
            "selected": selected, "created_unix": time.time(),
            "selection": f"Only fold {fold} development session {fold % 5 + 1}; seed17",
            "protocol": "3 train sessions / next cyclic dev session / held-out test session",
            "limitation": "Pilot data informed architecture development; this lock controls hyperparameter selection."}
    write_json(lock_path, lock)
    return lock, lock_path


def campaign(cache, output, device="cuda", pilot_only=True, finalize=False,
             max_epochs=100, patience=10, max_trials=12):
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    from .data import load_manifest
    manifest = load_manifest(cache)
    manifest_hash = digest(manifest)
    if max_trials < 2 or max_trials > 12:
        raise ValueError("max_trials must be between 2 and 12 (baseline plus graph required)")
    base = {"max_epochs": max_epochs, "patience": patience, "fold": 1, "seed": 17}
    menu = {"schema_version": 1, "manifest_hash": manifest_hash, "max_trials": max_trials,
            "plateau_stop": 4, "feature_hash": cache_fingerprint(cache, manifest), "final_seeds": [17, 42, 123],
            "candidates": {track: [[name, config] for name, config in candidate_configs(track, base)[:max_trials]]
                           for track in ("audio", "multi")}}
    menu_path = output / "candidate_menu.json"
    if menu_path.exists() and json.loads(menu_path.read_text()) != menu:
        raise ValueError("Candidate menu is immutable; use the same campaign settings or a new output directory")
    write_json(menu_path, menu)  # Predeclare before reading any development result.
    if pilot_only and not finalize:
        lock, path = screen_fold(cache, output, 1, menu, manifest_hash, device)
        print(f"Development selection locked in {path}; no test evaluation performed.", flush=True)
        return lock
    # Lock every fold independently before exposing any final test result.
    locks = {fold: screen_fold(cache, output, fold, menu, manifest_hash, device)
             for fold in range(1, 6)}
    final_rows = []
    for fold, (lock, lock_path) in locks.items():
        for name, choice in lock["selected"].items():
            for seed in menu["final_seeds"]:
                config = dict(choice["config"], fold=fold, seed=seed)
                # Seed17 already completed in this fold's screening; reuse that exact run.
                run = Path(choice["path"]) if seed == 17 else output / "final" / name / f"fold{fold}_seed{seed}"
                result = train(cache, run, config, device)
                test = json.loads((run / "test.json").read_text()) if (run / "test.json").exists() else evaluate(cache, run, "test", device, lock_path)
                final_rows.append({"model": name, "fold": fold, "seed": seed,
                                   "config": config, "dev_score": result["dev_score"], "test": test})
                write_json(output / "final_results.json", final_rows)
    summary = {}
    for name in locks[1][0]["selected"]:
        rows = [row for row in final_rows if row["model"] == name]
        scenarios = ("A",) if name.startswith("audio_") else tuple(SUBSETS)
        summary[name] = {}
        for scenario in scenarios:
            summary[name][scenario] = {}
            for metric in ("macro_f1", "uar", "weighted_f1", "accuracy"):
                values = [row["test"][scenario][metric] for row in rows]
                summary[name][scenario][metric] = {"mean": float(np.mean(values)),
                    "std": float(np.std(values)), "n_runs": len(values)}
    write_json(output / "final_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "campaign"):
        command = commands.add_parser(name)
        command.add_argument("--cache", required=True)
        command.add_argument("--device", default="cuda")
        if name == "evaluate":
            command.add_argument("--run", required=True)
            command.add_argument("--split", choices=("dev", "test"), default="dev")
            command.add_argument("--lock")
        else:
            command.add_argument("--output", required=True)
            command.add_argument("--max-epochs", type=int, default=100)
            command.add_argument("--patience", type=int, default=10)
        if name == "train":
            command.add_argument("--config", help="JSON overrides of the saved default configuration")
            command.add_argument("--variant", choices=("audio_mlp", "audio_homo", "audio_uniform", "audio_graph", "fusion", "multi_graph"))
            command.add_argument("--fold", type=int, default=1)
            command.add_argument("--seed", type=int, default=17)
        if name == "campaign":
            action = command.add_mutually_exclusive_group()
            action.add_argument("--pilot-only", action="store_true")
            action.add_argument("--finalize", action="store_true")
            command.add_argument("--max-trials", type=int, default=12)
    args = parser.parse_args()
    if args.command == "train":
        config = json.loads(Path(args.config).read_text()) if args.config else {}
        config.update(fold=args.fold, seed=args.seed, max_epochs=args.max_epochs, patience=args.patience)
        if args.variant:
            config["variant"] = args.variant
        train(args.cache, args.output, config, args.device)
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.cache, args.run, args.split, args.device, args.lock), indent=2))
    else:
        campaign(args.cache, args.output, args.device, pilot_only=not args.finalize, finalize=args.finalize,
                 max_epochs=args.max_epochs, patience=args.patience, max_trials=args.max_trials)


if __name__ == "__main__":
    main()
