"""Memory-mapped sequence access for daily cross-market feature panels.

The store contains one row per eligible trading day, not one row per 60-day
window.  This is intentional: expanding the full panel would consume tens of
gigabytes before training starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class StoreMetadata:
    n_rows: int
    n_features: int
    feature_columns: list[str]
    domain_map: dict[str, int]
    files: dict[str, str]
    window_counts: dict[str, int]
    window_length: int


class SequenceStore:
    """Read-only access to a sequence store built by build_sequence_store.py."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        with (self.root / "metadata.json").open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.meta = StoreMetadata(**raw)
        self.features = np.memmap(
            self.root / self.meta.files["features"], dtype=np.float32,
            mode="r", shape=(self.meta.n_rows, self.meta.n_features),
        )
        self.dates = np.memmap(
            self.root / self.meta.files["dates"], dtype=np.int32,
            mode="r", shape=(self.meta.n_rows,),
        )
        self.domains = np.memmap(
            self.root / self.meta.files["domains"], dtype=np.uint8,
            mode="r", shape=(self.meta.n_rows,),
        )
        self.resume_flags = np.memmap(
            self.root / self.meta.files["resume_flags"], dtype=np.uint8,
            mode="r", shape=(self.meta.n_rows,),
        )
        self.offsets = pd.read_parquet(self.root / self.meta.files["offsets"])
        self._window_cache: dict[str, np.memmap] = {}

    @property
    def market_to_domain(self) -> dict[str, int]:
        return self.meta.domain_map

    def windows(self, kind: str) -> np.memmap:
        if kind not in self.meta.window_counts:
            raise ValueError(f"Unknown window kind {kind!r}; choose {list(self.meta.window_counts)}")
        if kind not in self._window_cache:
            self._window_cache[kind] = np.memmap(
                self.root / self.meta.files[f"windows_{kind}"], dtype=np.int64,
                mode="r", shape=(self.meta.window_counts[kind],),
            )
        return self._window_cache[kind]

    def select_window_ends(
        self, kind: str, markets: Iterable[str], years: Iterable[int], max_items: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Return endpoint row indexes whose endpoint belongs to market/year."""
        market_ids = np.array([self.market_to_domain[m] for m in markets], dtype=np.uint8)
        years_arr = np.asarray(list(years), dtype=np.int32)
        endpoints = np.asarray(self.windows(kind))
        endpoint_dates = np.asarray(self.dates[endpoints])
        endpoint_years = endpoint_dates // 10000
        endpoint_domains = np.asarray(self.domains[endpoints])
        selected = endpoints[np.isin(endpoint_domains, market_ids) & np.isin(endpoint_years, years_arr)]
        if max_items is not None and len(selected) > max_items:
            rng = np.random.default_rng(seed)
            selected = rng.choice(selected, size=max_items, replace=False)
        return np.asarray(selected, dtype=np.int64)

    def endpoint_symbols(self, endpoints: np.ndarray) -> np.ndarray:
        """Map endpoint rows to symbols using sorted symbol segment offsets."""
        starts = self.offsets["start"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(starts, endpoints, side="right") - 1
        if np.any(positions < 0):
            raise ValueError("Endpoint before first symbol offset")
        return self.offsets["symbol"].to_numpy()[positions]


def fit_market_scalers(
    store: SequenceStore, endpoints: np.ndarray, markets: Iterable[str], *, sample_limit: int,
    seed: int, mad_constant: float = 1.4826, epsilon: float = 1e-6,
) -> dict[str, dict[str, list[float]]]:
    """Fit robust, market-specific scalers using *training endpoints only*."""
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, list[float]]] = {}
    endpoint_domains = np.asarray(store.domains[endpoints])
    for market in markets:
        chosen = endpoints[endpoint_domains == store.market_to_domain[market]]
        if len(chosen) == 0:
            raise ValueError(f"No training windows available for market={market}")
        if len(chosen) > sample_limit:
            chosen = rng.choice(chosen, size=sample_limit, replace=False)
        x = np.asarray(store.features[chosen], dtype=np.float32)
        median = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - median), axis=0) * mad_constant
        scale = np.maximum(mad, epsilon)
        result[market] = {"median": median.tolist(), "scale": scale.tolist()}
    return result


class WindowDataset(Dataset):
    """Returns normalized [window, feature] tensors from legal endpoint rows."""

    def __init__(
        self, store: SequenceStore, endpoints: np.ndarray, scalers: dict[str, dict[str, list[float]]],
        *, clip: tuple[float, float] = (-8.0, 8.0), return_endpoint: bool = False,
    ):
        self.store = store
        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.return_endpoint = return_endpoint
        self.clip = clip
        self.window_length = store.meta.window_length
        self.domain_ids = np.asarray(store.domains[self.endpoints], dtype=np.int64)
        self._median = {
            store.market_to_domain[name]: np.asarray(value["median"], dtype=np.float32)
            for name, value in scalers.items()
        }
        self._scale = {
            store.market_to_domain[name]: np.asarray(value["scale"], dtype=np.float32)
            for name, value in scalers.items()
        }

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, idx: int):
        end = int(self.endpoints[idx])
        domain = int(self.domain_ids[idx])
        start = end - self.window_length + 1
        x = np.array(self.store.features[start:end + 1], dtype=np.float32, copy=True)
        x = (x - self._median[domain]) / self._scale[domain]
        x = np.clip(x, self.clip[0], self.clip[1])
        answer = (torch.from_numpy(x), torch.tensor(domain, dtype=torch.long))
        if self.return_endpoint:
            return answer + (torch.tensor(end, dtype=torch.long),)
        return answer


class BalancedDomainBatchSampler(Sampler[list[int]]):
    """Infinite-within-epoch replacement sampler with declared market mix."""

    def __init__(
        self, dataset: WindowDataset, market_mix: dict[str, float], batch_size: int,
        steps_per_epoch: int, seed: int,
    ):
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        raw_counts = {m: batch_size * w for m, w in market_mix.items()}
        counts = {m: int(np.floor(v)) for m, v in raw_counts.items()}
        remainder = batch_size - sum(counts.values())
        for market, _ in sorted(raw_counts.items(), key=lambda kv: kv[1] - np.floor(kv[1]), reverse=True)[:remainder]:
            counts[market] += 1
        self.positions = {
            dataset.store.market_to_domain[m]: np.flatnonzero(dataset.domain_ids == dataset.store.market_to_domain[m])
            for m in market_mix
        }
        if any(len(self.positions[d]) == 0 for d in self.positions):
            raise ValueError("A requested market has no training windows")
        self.counts = {dataset.store.market_to_domain[m]: n for m, n in counts.items()}

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps_per_epoch):
            batch = np.concatenate([
                rng.choice(self.positions[domain], size=count, replace=True)
                for domain, count in self.counts.items() if count > 0
            ])
            rng.shuffle(batch)
            yield batch.tolist()
