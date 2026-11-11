"""Dialogue-local, fixed-metapath classifiers with explicit observation masks.

No graph, speaker embedding, or normalization statistic is shared across test
dialogues. Inputs are frozen utterance features; normalization belongs to the
trainer. ``set_prosody_prior`` must use that same input coordinate system.
"""
import torch
from torch import nn


MODALITIES = ("audio", "text", "visual")


def _nearest_edges(allowed, distance, max_neighbors):
    """Return [source, target] indices, at most K closest sources per target."""
    n = allowed.shape[0]
    rank = distance.masked_fill(~allowed, torch.inf).argsort(dim=1, stable=True)
    selected = torch.zeros_like(allowed)
    selected.scatter_(1, rank[:, :min(max_neighbors, n)], True)
    target, source = (allowed & selected).nonzero(as_tuple=True)
    return torch.stack((source, target))


def build_paths(batch, audio_mask=None, window=8, max_neighbors=8):
    """Fixed MP1 U-S-U, MP2 same-speaker U-P-P-U, MP3 response U-P-P-U.

    Both temporal directions are allowed in this offline dialogue task. MP2
    links consecutive turns of a known speaker; MP3 links adjacent turns of
    different known speakers. Missing nodes remove original path instances;
    they do not create new adjacency by skipping an unobserved turn.
    """
    speaker = batch["speaker"]
    n = speaker.numel()
    order = batch.get("order", torch.arange(n, device=speaker.device))
    if n == 0 or order.unique().numel() != n:
        raise ValueError("A dialogue needs at least one uniquely ordered utterance")
    delta = order[None, :] - order[:, None]
    distance = delta.abs().float()
    known = (speaker[:, None] >= 0) & (speaker[None, :] >= 0)
    same = known & (speaker[:, None] == speaker[None, :])
    bounded = (distance > 0) & (distance <= window)
    same_other = same & (distance > 0)
    following = distance.masked_fill(~(same_other & (delta > 0)), torch.inf)
    preceding = distance.masked_fill(~(same_other & (delta < 0)), torch.inf)
    consecutive = ((following == following.min(1).values[:, None]) & (delta > 0)
                   | (preceding == preceding.min(1).values[:, None]) & (delta < 0))
    positions = torch.empty_like(order)
    positions[order.argsort()] = torch.arange(n, device=order.device)
    adjacent = (positions[:, None] - positions[None, :]).abs() == 1
    allowed = {1: same & bounded, 2: same & bounded & consecutive,
               3: known & ~same & adjacent & bounded}
    edges = {key: _nearest_edges(value, distance, max_neighbors)
             for key, value in allowed.items()}
    if audio_mask is not None:
        edges = {key: pair[:, audio_mask[pair[0]] & audio_mask[pair[1]]]
                 for key, pair in edges.items()}
    return edges


def _masked_softmax(scores, available, dim):
    weights = scores.masked_fill(~available, torch.finfo(scores.dtype).min).softmax(dim)
    weights = weights * available.to(weights.dtype)
    return weights / weights.sum(dim, keepdim=True).clamp_min(1e-12)


def _speaker_pool(features, speaker, observed):
    same = ((speaker[:, None] == speaker[None, :])
            & (speaker[:, None] >= 0) & observed[None, :])
    weights = same.to(features.dtype)
    return weights @ features / weights.sum(1, keepdim=True).clamp_min(1)


class _PathMessage(nn.Module):
    def __init__(self, hidden, intermediates, attention, dropout):
        super().__init__()
        self.message = nn.Linear((1 + intermediates) * hidden, hidden)
        self.relation = nn.Linear(4, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def forward(self, source, target, edges, relation, intermediate):
        n, hidden = target.shape
        src, dst = edges
        valid = torch.zeros(n, dtype=torch.bool, device=target.device)
        if src.numel() == 0:
            return torch.zeros_like(target), valid
        valid[dst] = True
        inputs = [source[src], *intermediate]
        messages = torch.nn.functional.gelu(self.message(torch.cat(inputs, -1))
                                             + self.relation(relation))
        if self.attention:
            score = self.score(torch.tanh(messages + self.query(target[dst]))).squeeze(-1)
            maximum = score.new_full((n,), -torch.inf)
            maximum.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)
            weight = (score - maximum[dst]).exp()
        else:
            weight = messages.new_ones(src.numel())
        denominator = weight.new_zeros(n).index_add_(0, dst, weight)
        weight = weight / denominator[dst].clamp_min(1e-12)
        result = target.new_zeros(n, hidden)
        result.index_add_(0, dst, self.dropout(messages) * weight[:, None])
        return result, valid


class _GraphLayer(nn.Module):
    def __init__(self, hidden, relations, attention, dropout):
        super().__init__()
        self.messages = nn.ModuleDict({name: _PathMessage(hidden, mids, attention, dropout)
                                       for name, mids in relations.items()})
        self.semantic = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.Tanh(),
                                      nn.Linear(hidden // 2, 1, bias=False))
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.dropout = nn.Dropout(dropout)
        self.attention = attention

    def finish(self, state, summaries, validity, observation=None):
        if summaries:
            values = torch.stack(summaries, 1)
            available = torch.stack(validity, 1)
            if self.attention:
                query = state[:, None, :].expand_as(values)
                scores = self.semantic(torch.cat((query, values), -1)).squeeze(-1)
            else:
                scores = values.new_zeros(available.shape)
            weights = _masked_softmax(scores, available, 1)
            aggregate = (values * weights[:, :, None]).sum(1)
        else:
            aggregate = torch.zeros_like(state)
        if hasattr(self, "context_gate"):
            inputs = [state, aggregate, (state - aggregate).abs()]
            if self.context_gate_observed:
                if observation is None:
                    raise ValueError("Observation-aware context gate requires observation descriptors")
                inputs.append(observation)
            # Identity at initialization, with a learned total message gain in (0, 2).
            aggregate = aggregate * (2 * self.context_gate(torch.cat(inputs, -1)).sigmoid())
        state = self.norm1(state + self.dropout(aggregate))
        return self.norm2(state + self.dropout(self.ffn(state)))


class InductiveGraph(nn.Module):
    """Six variants sharing a dialogue-local classifier interface.

    ``audio_mlp`` uses SSL audio only. ``audio_homo`` uses audio and prosody
    with untyped local neighbors. ``audio_uniform`` uses the fixed paths with
    uniform instance/path weights and raw prosody by default. ``audio_graph``
    learns those weights and defaults to raw plus speaker-relative prosody.
    ``fusion`` independently encodes modalities and uses a masked gate;
    ``multi_graph`` additionally exchanges only observed graph messages.
    """
    def __init__(self, dims, num_classes=6, variant="audio_graph", hidden=256,
                 layers=2, dropout=.2, **options):
        super().__init__()
        variants = ("audio_mlp", "audio_homo", "audio_uniform", "audio_graph",
                    "fusion", "multi_graph")
        if variant not in variants:
            raise ValueError(f"Unknown variant: {variant}")
        allowed_options = {"paths", "relative", "intermediate", "path_attention",
                           "window", "max_neighbors", "prior_strength", "context_gate"}
        if set(options) - allowed_options:
            raise ValueError(f"Unknown model options: {set(options) - allowed_options}")
        self.variant = variant
        self.audio_only = variant.startswith("audio_")
        self.relative = options.get("relative", variant not in ("audio_mlp", "audio_uniform"))
        self.intermediate = options.get("intermediate", True)
        self.path_attention = options.get("path_attention", variant != "audio_uniform")
        self.paths = tuple(options.get("paths", (1, 2, 3)))
        if not set(self.paths).issubset({1, 2, 3}) or len(set(self.paths)) != len(self.paths):
            raise ValueError("paths must be a unique subset of (1, 2, 3)")
        self.window = options.get("window", 8)
        self.max_neighbors = options.get("max_neighbors", 8)
        self.prior_strength = options.get("prior_strength", 2.)
        if self.window <= 0 or self.max_neighbors <= 0 or self.prior_strength <= 0:
            raise ValueError("window, max_neighbors and prior_strength must be positive")
        if hidden < 2 or layers < 0:
            raise ValueError("hidden >= 2 and layers >= 0 are required")
        self.dims = dict(dims)
        self.hidden = hidden
        names = ("audio",) if self.audio_only else MODALITIES
        self.projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(dims[name], hidden),
                                                 nn.LayerNorm(hidden), nn.GELU()) for name in names})
        self.use_prosody = variant != "audio_mlp"
        prosody_dim = dims.get("prosody", 16)
        self.register_buffer("prosody_mean", torch.zeros(prosody_dim))
        self.register_buffer("prosody_std", torch.ones(prosody_dim))
        if self.use_prosody:
            self.prosody_projection = nn.Sequential(
                nn.Linear(prosody_dim * (2 if self.relative else 1), hidden),
                nn.LayerNorm(hidden), nn.GELU())
        relations = {}
        if variant in ("audio_uniform", "audio_graph", "multi_graph"):
            relations.update({f"a{path}": (1 if path == 1 else 2) if self.intermediate else 0
                              for path in self.paths})
        if variant == "audio_homo":
            relations["homo"] = 0
        if variant == "multi_graph":
            relations.update({"t1": int(self.intermediate), "v1": int(self.intermediate)})
            relations.update({f"cross{src}{dst}": 0 for src in range(3) for dst in range(3)
                              if src != dst})
        self.layers = nn.ModuleList([_GraphLayer(hidden, relations, self.path_attention, dropout)
                                     for _ in range(layers)])
        self.modality_gate = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout),
                                         nn.Linear(hidden, num_classes))
        self.context_gate_mode = options.get("context_gate", "none")
        if self.context_gate_mode not in ("none", "state", "observed"):
            raise ValueError("context_gate must be none, state, or observed")
        if self.context_gate_mode != "none":
            # Preserve shared parameter initialization and the training dropout RNG.
            with torch.random.fork_rng(devices=[]):
                for layer in self.layers:
                    layer.context_gate_observed = self.context_gate_mode == "observed"
                    extra = 7 if layer.context_gate_observed else 0
                    layer.context_gate = nn.Sequential(nn.Linear(3 * hidden + extra, 32),
                                                        nn.GELU(), nn.Linear(32, 1))
                    nn.init.zeros_(layer.context_gate[-1].weight)
                    nn.init.zeros_(layer.context_gate[-1].bias)

    def set_prosody_prior(self, mean, std):
        mean = torch.as_tensor(mean, device=self.prosody_mean.device, dtype=self.prosody_mean.dtype)
        std = torch.as_tensor(std, device=self.prosody_std.device, dtype=self.prosody_std.dtype)
        if mean.shape != self.prosody_mean.shape or std.shape != self.prosody_std.shape:
            raise ValueError("Prosody prior shape does not match prosody dimension")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Prosody prior must be finite with positive standard deviations")
        with torch.no_grad():
            self.prosody_mean.copy_(mean)
            self.prosody_std.copy_(std)

    def _observed_input(self, batch, name, observed, reference):
        if name not in batch:
            if observed.any():
                raise ValueError(f"Observed modality {name} is absent from batch")
            return reference.new_zeros(observed.numel(), self.dims[name])
        value = batch[name]
        expected = (observed.numel(), self.dims[name])
        if value.shape != expected:
            raise ValueError(f"{name} shape {tuple(value.shape)} differs from {expected}")
        value = torch.where(observed[:, None], value, torch.zeros_like(value))
        if not torch.isfinite(value).all():
            raise ValueError(f"Observed {name} features contain NaN or Inf")
        return value

    def _prosody(self, raw, speaker, observed):
        if self.relative:
            same = ((speaker[:, None] == speaker[None, :])
                    & (speaker[:, None] >= 0) & observed[None, :])
            weights = same.to(raw.dtype)
            strength = self.prior_strength
            count = weights.sum(1, keepdim=True) + strength
            mean = (weights @ raw + strength * self.prosody_mean) / count
            moment = (weights @ raw.square()
                      + strength * (self.prosody_std.square() + self.prosody_mean.square())) / count
            variance = (moment - mean.square()).clamp_min(1e-4)
            relative = (raw - mean) / variance.sqrt()
            raw = torch.cat((raw, relative), -1)
        projected = self.prosody_projection(raw)
        return projected * observed[:, None]

    def _relation(self, edges, batch, dtype):
        src, dst = edges
        order = batch["order"]
        delta = (order[src] - order[dst]).to(dtype) / self.window
        if "times" in batch:
            times = batch["times"].mean(1)
            seconds = torch.tanh((times[src] - times[dst]).to(dtype) / 10.)
        else:
            seconds = torch.zeros_like(delta)
        return torch.stack((delta, delta.abs(), seconds, seconds.abs()), -1)

    @staticmethod
    def _keep_features(value, name, feature_keep):
        if feature_keep is None or name not in feature_keep:
            return value
        keep = feature_keep[name]
        if keep.shape != value.shape or keep.dtype != torch.bool or keep.device != value.device:
            raise ValueError(f"Feature keep mask for {name} must be bool with matching shape/device")
        return value * keep

    @staticmethod
    def _keep_edges(edges, name, edge_keep, n):
        if edge_keep is None or name not in edge_keep:
            return edges
        keep = edge_keep[name]
        if keep.shape != (n, n) or keep.dtype != torch.bool or keep.device != edges.device:
            raise ValueError(f"Edge keep mask for {name} must be bool[N,N] on the graph device")
        return edges[:, keep[edges[0], edges[1]]]

    def forward(self, batch, mask=None):
        return self.classifier(self.encode(batch, mask))

    def encode(self, batch, mask=None, *, feature_keep=None, edge_keep=None):
        """Pre-classifier utterance states; optional GCL masks only remove information.

        No masks means the original inference/training computation. Graph views
        filter existing relations without rebuilding adjacency after deletion.
        """
        reference = next(batch[name] for name in MODALITIES if name in batch)
        n, device = reference.shape[0], reference.device
        if n == 0:
            raise ValueError("Empty dialogues are not supported")
        if mask is None:
            mask = torch.ones((n, 3), dtype=torch.bool, device=device)
        else:
            if mask.shape != (n, 3) or mask.dtype != torch.bool:
                raise ValueError("mask must be bool[N,3] in audio,text,visual order")
            mask = mask.to(device)
        if self.audio_only:
            if not mask[:, 0].all():
                raise ValueError("Audio-only variants require observed audio for every utterance")
            mask = mask.clone()
            mask[:, 1:] = False
        if not mask.any(1).all():
            raise ValueError("Every utterance needs at least one observed modality")
        speaker = batch.get("speaker", torch.full((n,), -1, device=device, dtype=torch.long))
        order = batch.get("order", torch.arange(n, device=device))
        if speaker.shape != (n,) or order.shape != (n,):
            raise ValueError("speaker and order must be one-dimensional utterance arrays")
        local = {**batch, "speaker": speaker, "order": order}
        states = []
        for m, name in enumerate(MODALITIES):
            if name not in self.projections:
                states.append(reference.new_zeros(n, self.hidden))
                continue
            value = self._observed_input(batch, name, mask[:, m], reference)
            value = self._keep_features(value, name, feature_keep)
            states.append(self.projections[name](value) * mask[:, m, None])
        prosody = reference.new_zeros(n, self.hidden)
        if self.use_prosody:
            raw = self._observed_input(batch, "prosody", mask[:, 0], reference)
            raw = self._keep_features(raw, "prosody", feature_keep)
            prosody = self._prosody(raw, speaker, mask[:, 0])
            states[0] = states[0] + prosody
        paths = build_paths(local, window=self.window, max_neighbors=self.max_neighbors)
        if self.variant == "audio_homo":
            distance = (order[:, None] - order[None, :]).abs().float()
            paths[0] = _nearest_edges((distance > 0) & (distance <= self.window),
                                       distance, self.max_neighbors)
        for layer in self.layers:
            summaries, validity = [[] for _ in range(3)], [[] for _ in range(3)]
            counts = [reference.new_zeros(n) for _ in range(3)] if self.context_gate_mode == "observed" else None
            for name, message in layer.messages.items():
                if name.startswith("cross"):
                    src_m, dst_m = int(name[-2]), int(name[-1])
                    observed = (mask[:, src_m] & mask[:, dst_m]).nonzero().flatten()
                    edges = torch.stack((observed, observed))
                    edges = self._keep_edges(edges, name, edge_keep, n)
                    middle = []
                else:
                    dst_m = src_m = {"a": 0, "t": 1, "v": 2, "h": 0}[name[0]]
                    path = 0 if name == "homo" else int(name[-1])
                    edges = paths[path]
                    edges = edges[:, mask[edges[0], src_m] & mask[edges[1], dst_m]]
                    edges = self._keep_edges(edges, name, edge_keep, n)
                    source, target = edges
                    middle = []
                    if self.intermediate and path == 1:
                        speaker_input = states[src_m] + (prosody if src_m == 0 else 0)
                        pooled = _speaker_pool(speaker_input, speaker, mask[:, src_m])
                        middle = [pooled[target]]
                    elif self.intermediate and path in (2, 3):
                        middle = [prosody[source], prosody[target]]
                relation = self._relation(edges, local, reference.dtype)
                values, valid = message(states[src_m], states[dst_m], edges, relation, middle)
                summaries[dst_m].append(values)
                validity[dst_m].append(valid)
                if counts is not None:
                    counts[dst_m].index_add_(0, edges[1], reference.new_ones(edges.shape[1]))
            observations = [None] * 3
            if counts is not None:
                # Only actual observed relation instances; repeated paths count separately.
                count_features = torch.stack(counts, 1).log1p()
                for m in range(3):
                    valid_fraction = (torch.stack(validity[m], 1).to(reference.dtype).mean(1)
                                      if validity[m] else reference.new_zeros(n))
                    observations[m] = torch.cat((mask.to(reference.dtype), count_features,
                                                  valid_fraction[:, None]), -1)
            states = [layer.finish(state, summaries[m], validity[m], observations[m]) * mask[:, m, None]
                      if (not self.audio_only or m == 0) else state
                      for m, state in enumerate(states)]
        if self.audio_only:
            fused = states[0]
        else:
            stacked = torch.stack(states, 1)
            gate = _masked_softmax(self.modality_gate(stacked).squeeze(-1), mask, 1)
            fused = (stacked * gate[:, :, None]).sum(1)
        return fused
