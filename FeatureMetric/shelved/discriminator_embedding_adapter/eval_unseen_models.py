"""Unseen-model generalization: AUC(ERA5 vs forecast) for each trained
discriminator across seen and unseen forecast models at 24h lead.

Both checkpoints were trained on Pangu+GraphCast only, so FuXi (unseen NN),
IFS-HRES and ERA5-Forecast (unseen numerical) are true holdouts. For NN models
a good realism metric wants HIGH AUC (detects smoothing); for numerical models
it wants LOW / real-like AUC (they are physically realistic, should not be
flagged). Compares raw vs frozen-embedding discriminators on that axis.
"""
import os, sys
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from torch.utils.data import DataLoader
from hydra import initialize_config_dir, compose

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_discriminator import WeatherDiscriminatorDataset, variables_from_config
from analysis_utils import load_discriminator
from embedding_adapter import load_field_stats_sidecar

CONFDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conf")
DATA = os.environ["DATA_DIR"]
REAL = f"{DATA}/era5-gt_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"
MODELS = [  # (label, file, kind)
    ("Pangu",         f"{DATA}/pangu_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc",        "seen NN"),
    ("GraphCast",     f"{DATA}/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc",    "seen NN"),
    ("FuXi",          f"{DATA}/fuxi_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc",         "UNSEEN NN"),
    ("IFS-HRES",      f"{DATA}/ifs_hres_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc",     "UNSEEN phys"),
    ("ERA5-Forecast", f"{DATA}/era5_forecast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc","UNSEEN phys"),
]
CKPTS = [  # (label, filename, is_raw)
    ("raw",         "weather_discriminator_raw_squeezenet_v2_all_fields_lightning.pth", True),
    ("embed+corr",  "weather_discriminator_embed_sfno_squeezenet_aug_all_fields_lightning.pth", False),
    ("embed-clean", "weather_discriminator_embed_sfno_squeezenet_noaug_all_fields_lightning.pth", False),
]
MAXN = int(os.environ.get("EVAL_MAX", "400"))


def model_logits(model, ds, device):
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).reshape(-1).cpu().numpy())
            labels.append(np.asarray(y).reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for clabel, fname, is_raw in CKPTS:
        ov = ["++model_name=squeezenet"] + (["++embedding_encoder=null"] if is_raw else [])
        with initialize_config_dir(version_base=None, config_dir=CONFDIR):
            cfg = compose(config_name="embedding_config", overrides=ov)
        vars_ = variables_from_config(cfg)
        model, path = load_discriminator(cfg, vars_, device, filename=fname)
        means, stds = load_field_stats_sidecar(path)
        results[clabel] = {}
        for mlabel, mfile, kind in MODELS:
            ds = WeatherDiscriminatorDataset(
                REAL, mfile, vars_, real_range=["2020-01-01", "2020-12-31"],
                fake_range=["2020-01-01", "2020-12-31"], lead_times=[24],
                level=cfg.get("level"), balanced=False, means=means, stds=stds,
                max_samples=MAXN,
            )
            lg, lab = model_logits(model, ds, device)
            r, f = lg[lab == 1.0], lg[lab == 0.0]
            u, _ = mannwhitneyu(r, f, alternative="greater")
            auc = float(u) / (len(r) * len(f))
            results[clabel][mlabel] = (auc, float(r.mean()), float(f.mean()))
            print(f"{clabel:<12} {mlabel:<14} ({kind:<11}) AUC={auc:.3f}  "
                  f"mean_logit ERA5={r.mean():+.2f} model={f.mean():+.2f}  "
                  f"(>0 = real-like)  n={len(r)}", flush=True)

    print("\n=== AUC(ERA5 vs model)  [and mean forecast logit; >0 = scored real-like] ===")
    print(f"{'model':<16}{'kind':<13}" + "".join(f"{c:<20}" for c, _, _ in CKPTS))
    for mlabel, _, kind in MODELS:
        row = "".join(f"AUC={results[c][mlabel][0]:.2f} L={results[c][mlabel][2]:+.2f}   "
                      for c, _, _ in CKPTS)
        print(f"{mlabel:<16}{kind:<13}{row}")


if __name__ == "__main__":
    main()
