"""Frozen-embedding probe: can a linear head separate ERA5 from 24h forecasts?

Phase-A gate for the embedding-discriminator direction. Extracts per-patch
tokens from frozen encoders (MAE / I-JEPA / SFNO), fits capacity-controlled
probes (logistic regression, optional 2-layer MLP) on several feature readouts
(mean-pool, max-pool, mean||max concat, flat tokens), and reports test AUC
against the classical high-wavenumber spectral baseline on the same samples.

Split convention (matches the discriminator paper): train pairs come from
2018 (Pangu/GraphCast forecasts + valid-time-paired ERA5), test pairs from
2020 only. Guards assert the years so 2020 can never leak into training.

Usage (local, from FeatureMetric/):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_embedding_probe.py
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_embedding_probe.py \
        --encoders mae --n-train 64 --n-test 64        # smoke run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import hashlib

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.model_io import build_model, load_model_checkpoint
from psd_diagnostic import paired_indices, radial_psd, read_interior, _read_times

CHANNELS = ["T2M", "U10", "V10", "MSL"]
FEATURES = ("mean", "max", "concat", "tokens")

DEFAULT_ERA5 = "data/test_data_local_5y.nc"
DEFAULT_FORECASTS = {
    "pangu": {"train": "data/pangu_surface_2018_lead24h.nc",
              "test": "data/pangu_surface_2020_lead24h.nc"},
    "graphcast": {"train": "data/graphcast_surface_2018_lead24h.nc",
                  "test": "data/graphcast_surface_2020_lead24h.nc"},
}


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def load_encoder(name, args, device):
    """Build a frozen encoder and return (model, extract_fn) where extract_fn
    maps a NetCDF path + absolute time indices to a (N, n_tokens, D) array."""
    if name in ("mae", "ijepa"):
        ckpt_path = args.mae_checkpoint if name == "mae" else args.ijepa_checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        if cfg.get("temporal_mode", "none") != "none":
            raise SystemExit(f"{ckpt_path} is a temporal checkpoint; probe needs temporal_mode=none")
        model = build_model(
            name, device, cfg.get("model_size", "twin"),
            embed_dim=cfg.get("embed_dim"), num_heads=cfg.get("num_heads"),
            depth=cfg.get("depth"),
        )
        load_model_checkpoint(name, model, ckpt_path, device)
        model.eval()
        stats = (np.load(Path(args.stats_dir) / "data_mean.npy"),
                 np.load(Path(args.stats_dir) / "data_std.npy"))

        def extract(nc_path, indices):
            ds = AtmosphereDataset(nc_path, split="all", stats=stats, lazy=True)
            loader = DataLoader(Subset(ds, list(indices)), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
            out = []
            with torch.no_grad():
                for batch in loader:
                    tok = model.extract_patch_tokens(batch.to(device))
                    out.append(tok.cpu().numpy())
            return np.concatenate(out, axis=0)          # (N, 128, D)

        return extract

    if name == "sfno":
        from utils.sfno_embedding import SFNOEmbedding, RawFourVarDataset
        sfno = SFNOEmbedding(
            embedding_channels=args.sfno_channels,
            embedding_resolution=tuple(args.sfno_resolution),
        )
        sfno.eval()

        def extract(nc_path, indices):
            ds = RawFourVarDataset(nc_path)
            loader = DataLoader(Subset(ds, list(indices)), batch_size=args.batch_size,
                                shuffle=False, num_workers=0)
            out = []
            with torch.no_grad():
                for batch in loader:                    # SFNO: CPU, fp32, raw units
                    emb = sfno.encode(batch)            # (B, C, h, w)
                    b, c, h, w = emb.shape
                    out.append(emb.reshape(b, c, h * w).permute(0, 2, 1).numpy())
            return np.concatenate(out, axis=0)          # (N, h*w, C)

        return extract

    raise SystemExit(f"unknown encoder {name!r}")


def cached_tokens(cache_dir, encoder, nc_path, indices, extract_fn):
    """Extract (or load cached) per-token embeddings for one file + index set."""
    idx_hash = hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:10]
    cache = Path(cache_dir) / f"{encoder}_{Path(nc_path).stem}_{idx_hash}.npz"
    if cache.exists():
        return np.load(cache)["tokens"]
    tokens = extract_fn(nc_path, indices).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tokens=tokens)
    print(f"  cached {cache.name}  {tokens.shape}")
    return tokens


def features_from_tokens(tokens):
    """(N, n_tok, D) -> {feature_name: (N, dim)} readouts."""
    mean = tokens.mean(axis=1)
    mx = tokens.max(axis=1)
    return {
        "mean": mean,
        "max": mx,
        "concat": np.concatenate([mean, mx], axis=1),
        "tokens": tokens.reshape(tokens.shape[0], -1),
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _fit_probe(head, Xtr, ytr, Xte, seed, steps=500, lr=1e-2, weight_decay=1e-4):
    torch.manual_seed(seed)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr_t = torch.from_numpy((Xtr - mu) / sd)
    Xte_t = torch.from_numpy((Xte - mu) / sd)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).unsqueeze(1)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    head.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(head(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return head(Xte_t).squeeze(1).numpy()


def fit_logistic_probe(Xtr, ytr, Xte, seed):
    return _fit_probe(nn.Linear(Xtr.shape[1], 1), Xtr, ytr, Xte, seed)


def fit_mlp_probe(Xtr, ytr, Xte, seed):
    head = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1),
    )
    return _fit_probe(head, Xtr, ytr, Xte, seed)


PROBES = {"logistic": fit_logistic_probe, "mlp": fit_mlp_probe}


def auc_from_scores(scores_real, scores_fake):
    """AUC = P(score_real > score_fake) via the Mann-Whitney U statistic."""
    u, _ = mannwhitneyu(scores_real, scores_fake, alternative="greater")
    return float(u) / (len(scores_real) * len(scores_fake))


# ---------------------------------------------------------------------------
# Spectral baseline (same estimator as psd_diagnostic.py, same test samples)
# ---------------------------------------------------------------------------

def per_sample_highk(fields_arr, channel=0, k_lo=0.25):
    out = []
    for n in range(fields_arr.shape[0]):
        kk, E = radial_psd(fields_arr[n:n + 1, channel:channel + 1])
        out.append(E[0, kk >= k_lo].sum())
    return np.array(out)


def spectral_baseline_auc(era5_path, fc_path, e_idx, f_idx, stats):
    e_bp = per_sample_highk(read_interior(era5_path, stats, e_idx))
    f_bp = per_sample_highk(read_interior(fc_path, stats, f_idx))
    return auc_from_scores(e_bp, f_bp)


# ---------------------------------------------------------------------------
# Split construction + guards
# ---------------------------------------------------------------------------

def build_split(era5_path, fc_path, expected_year, n_cap, rng):
    """Valid-time-paired (era5_indices, forecast_indices), year-guarded."""
    e_idx, f_idx = paired_indices(era5_path, fc_path)
    if not e_idx:
        raise SystemExit(f"no paired valid times between {era5_path} and {fc_path}")
    era5_times, fc_times = _read_times(era5_path), _read_times(fc_path)
    for i in e_idx:
        assert era5_times[i].year == expected_year, \
            f"ERA5 sample {era5_times[i]} outside expected year {expected_year}"
    for i in f_idx:
        assert fc_times[i].year == expected_year, \
            f"forecast sample {fc_times[i]} outside expected year {expected_year}"
    if n_cap and len(e_idx) > n_cap:
        sel = rng.permutation(len(e_idx))[:n_cap]
        e_idx = [e_idx[i] for i in sel]
        f_idx = [f_idx[i] for i in sel]
    return e_idx, f_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["mae", "ijepa", "sfno"],
                        choices=["mae", "ijepa", "sfno"])
    parser.add_argument("--mae-checkpoint", default="checkpoints/best_mae_model_twin.pth")
    parser.add_argument("--ijepa-checkpoint", default="checkpoints/best_ijepa_model_tiny.pth")
    parser.add_argument("--sfno-channels", type=int, default=8)
    parser.add_argument("--sfno-resolution", type=int, nargs=2, default=[31, 60])
    parser.add_argument("--era5-path", default=DEFAULT_ERA5)
    parser.add_argument("--pangu-train", default=DEFAULT_FORECASTS["pangu"]["train"])
    parser.add_argument("--pangu-test", default=DEFAULT_FORECASTS["pangu"]["test"])
    parser.add_argument("--graphcast-train", default=DEFAULT_FORECASTS["graphcast"]["train"])
    parser.add_argument("--graphcast-test", default=DEFAULT_FORECASTS["graphcast"]["test"])
    parser.add_argument("--stats-dir", default="checkpoints",
                        help="Holds data_mean.npy / data_std.npy (ERA5 stats).")
    parser.add_argument("--cache-dir", default="results/probe_cache")
    parser.add_argument("--output-csv", default="results/embedding_probe_auc.csv")
    parser.add_argument("--probe", choices=["logistic", "mlp", "both"], default="logistic")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu",
                        help="Device for MAE/I-JEPA; SFNO always runs on CPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=0, help="0 = all paired samples")
    parser.add_argument("--n-test", type=int, default=0, help="0 = all paired samples")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Sanity control: permute train labels; AUC should be ~0.5.")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    stats = (np.load(Path(args.stats_dir) / "data_mean.npy"),
             np.load(Path(args.stats_dir) / "data_std.npy"))
    probes = ["logistic", "mlp"] if args.probe == "both" else [args.probe]

    forecasts = {
        "pangu": {"train": args.pangu_train, "test": args.pangu_test},
        "graphcast": {"train": args.graphcast_train, "test": args.graphcast_test},
    }

    # -- splits (shared across encoders) ------------------------------------
    splits = {}
    for fc_name, paths in forecasts.items():
        tr = build_split(args.era5_path, paths["train"], 2018, args.n_train, rng)
        te = build_split(args.era5_path, paths["test"], 2020, args.n_test, rng)
        assert not (set(tr[0]) & set(te[0])), "train/test ERA5 index overlap"
        splits[fc_name] = {"train": tr, "test": te}
        print(f"[{fc_name}] train pairs (2018): {len(tr[0])}   test pairs (2020): {len(te[0])}")

    rows = []

    # -- spectral baseline on the identical test samples --------------------
    if not args.skip_baseline:
        for fc_name, paths in forecasts.items():
            e_idx, f_idx = splits[fc_name]["test"]
            auc = spectral_baseline_auc(args.era5_path, paths["test"], e_idx, f_idx, stats)
            print(f"[baseline] spectral high-k (k>=0.25, T2M) vs {fc_name}: AUC={auc:.3f}")
            rows.append({"encoder": "spectral_highk", "forecast": fc_name, "feature": "-",
                         "probe": "-", "dim": 1, "n_train": 0,
                         "n_test": len(e_idx), "auc_test": auc, "acc_test": float("nan")})

    # -- probes --------------------------------------------------------------
    for enc_name in args.encoders:
        print(f"\n=== encoder: {enc_name} ===")
        extract = load_encoder(enc_name, args, device)
        for fc_name, paths in forecasts.items():
            sp = splits[fc_name]
            tok = {}
            for part, fc_path in (("train", paths["train"]), ("test", paths["test"])):
                e_idx, f_idx = sp[part]
                tok[part] = {
                    "real": cached_tokens(args.cache_dir, enc_name, args.era5_path, e_idx, extract),
                    "fake": cached_tokens(args.cache_dir, enc_name, fc_path, f_idx, extract),
                }

            feats_tr_real = features_from_tokens(tok["train"]["real"])
            feats_tr_fake = features_from_tokens(tok["train"]["fake"])
            feats_te_real = features_from_tokens(tok["test"]["real"])
            feats_te_fake = features_from_tokens(tok["test"]["fake"])

            for feat in FEATURES:
                Xtr = np.concatenate([feats_tr_real[feat], feats_tr_fake[feat]], axis=0)
                ytr = np.concatenate([np.ones(len(feats_tr_real[feat])),
                                      np.zeros(len(feats_tr_fake[feat]))])
                if args.shuffle_labels:
                    ytr = rng.permutation(ytr)
                Xte = np.concatenate([feats_te_real[feat], feats_te_fake[feat]], axis=0)
                n_te_real = len(feats_te_real[feat])

                for probe_name in probes:
                    scores = PROBES[probe_name](Xtr, ytr, Xte, args.seed)
                    s_real, s_fake = scores[:n_te_real], scores[n_te_real:]
                    auc = auc_from_scores(s_real, s_fake)
                    acc = float(((scores > 0) == np.concatenate(
                        [np.ones(n_te_real), np.zeros(len(s_fake))]).astype(bool)).mean())
                    rows.append({"encoder": enc_name, "forecast": fc_name, "feature": feat,
                                 "probe": probe_name, "dim": Xtr.shape[1],
                                 "n_train": len(ytr), "n_test": len(scores),
                                 "auc_test": auc, "acc_test": acc})
                    print(f"  [{fc_name}] {feat:<7} {probe_name:<9} dim={Xtr.shape[1]:<6} "
                          f"AUC={auc:.3f}  acc={acc:.3f}")

    # -- output ---------------------------------------------------------------
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    print("\n" + "=" * 78)
    print(f"{'encoder':<16}{'forecast':<12}{'feature':<9}{'probe':<10}"
          f"{'dim':>7}{'AUC':>8}{'acc':>8}")
    print("-" * 78)
    for r in rows:
        acc = f"{r['acc_test']:.3f}" if r["acc_test"] == r["acc_test"] else "-"
        print(f"{r['encoder']:<16}{r['forecast']:<12}{r['feature']:<9}{r['probe']:<10}"
              f"{r['dim']:>7}{r['auc_test']:>8.3f}{acc:>8}")
    print("=" * 78)


if __name__ == "__main__":
    main()
