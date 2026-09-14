"""Evaluate a trained discriminator and visualize spatial logit maps."""
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig
import xarray as xr

try:
    from .train_discriminator import WeatherDiscriminatorDataset, WeatherDiscriminator, variables_from_config
except ImportError:
    from train_discriminator import WeatherDiscriminatorDataset, WeatherDiscriminator, variables_from_config


PLOT_CONFIGS = {
    "u_component_of_wind": {"cmap": "viridis", "title": "u component of wind", "unit": "m/s"},
    "v_component_of_wind": {"cmap": "viridis", "title": "v component of wind", "unit": "m/s"},
    "wind_speed": {"cmap": "viridis", "title": "Wind Speed", "unit": "m/s"},
    "temperature": {"cmap": "inferno", "title": "Temperature", "unit": "K"},
    "2m_temperature": {"cmap": "inferno", "title": "2m Temperature", "unit": "K"},
    "10m_u_component_of_wind": {"cmap": "viridis", "title": "10m U Wind", "unit": "m/s"},
    "10m_v_component_of_wind": {"cmap": "viridis", "title": "10m V Wind", "unit": "m/s"},
    "specific_humidity": {"cmap": "GnBu", "title": "Specific Humidity", "unit": "kg/kg"},
    "mean_sea_level_pressure": {"cmap": "plasma", "title": "Mean Sea Level Pressure", "unit": "Pa"},
}


def default_weight_path(cfg, variables):
    """Match the filename written by `train_discriminator.py`."""
    variable_tag = cfg.get("selected_variable", "all_fields") if len(variables) == 1 else "all_fields"
    return os.path.join(cfg.output_dir, f"weather_discriminator_{cfg.model_name}_{variable_tag}_lightning.pth")


def validate_variables(real_nc_file, fake_nc_file, variables):
    """Check fields before constructing the heavier xarray-backed dataset."""
    with xr.open_dataset(real_nc_file) as real_ds, xr.open_dataset(fake_nc_file) as fake_ds:
        missing_real = [v for v in variables if v not in real_ds.data_vars]
        missing_fake = [v for v in variables if v not in fake_ds.data_vars]
    return missing_real, missing_fake


def run_test_inference(model, test_loader, test_dataset, device):
    """Return per-sample logits and aggregate classification accuracy."""
    results = []
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Inference")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += ((outputs > 0.0).float() == labels).sum().item()
            results.append({"idx": idx, "logit": outputs.item(), "label": labels.item()})

    return {
        "results": results,
        "accuracy": correct_predictions / len(test_dataset),
        "loss": running_loss / len(test_dataset),
    }


def sample_metadata(test_dataset, global_idx):
    """Map an unbalanced dataset index back to source dataset/time/lead metadata."""
    is_fake = global_idx >= test_dataset.total_real_samples
    if not is_fake:
        return {
            "ds": test_dataset.real_ds,
            "time": test_dataset.real_times[global_idx],
            "lead_idx": test_dataset.real_lead_idx,
            "data_type": "REAL",
        }

    fake_idx = global_idx - test_dataset.total_real_samples
    return {
        "ds": test_dataset.fake_ds,
        "time": test_dataset.fake_times[fake_idx // test_dataset.num_fake_leads],
        "lead_idx": test_dataset.fake_lead_indices[fake_idx % test_dataset.num_fake_leads],
        "data_type": "FAKE",
    }

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def evaluate_and_visualize(cfg: DictConfig):
    """Evaluate a trained discriminator and save representative map panels."""
    real_nc_file = cfg.test_real_nc_file
    fake_nc_file = cfg.test_fake_nc_file
    model_vars = variables_from_config(cfg)
    model_weights = default_weight_path(cfg, model_vars)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    print(f"Loading weights: {model_weights}")

    missing_real, missing_fake = validate_variables(real_nc_file, fake_nc_file, model_vars)
    if missing_real or missing_fake:
        print(f"Skipping evaluation: missing real variables={missing_real}; missing fake variables={missing_fake}.")
        return

    test_dataset = WeatherDiscriminatorDataset(
        real_nc_file, fake_nc_file, model_vars,
        real_range=cfg.test_real_ranges,
        fake_range=cfg.test_fake_range,
        lead_times=cfg.lead_times,
        level=cfg.get("level"),
        balanced=False,
        disturb_type=cfg.get("disturb_type", None),
        disturb_level=cfg.get("disturb_level", 0.0),
        monthly_split_cfg=cfg if cfg.get("monthly_split") else None,
        monthly_split_name="test",
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    lightning_model = WeatherDiscriminator(num_weather_channels=len(model_vars), model_name=cfg.model_name)
    lightning_model.model.load_state_dict(torch.load(model_weights, map_location=device))
    model = lightning_model.to(device).eval()

    metrics = run_test_inference(model, test_loader, test_dataset, device)
    results = metrics["results"]
    print(f"\nOverall Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall Test Loss: {metrics['loss']:.4f}")

    realest = sorted(results, key=lambda x: -x['logit'])[:3]
    uncertain = sorted(results, key=lambda x: abs(x['logit']))[:3]
    fooled = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'], reverse=True)[:3]
    obvious = sorted([r for r in results if r['label'] == 0.0], key=lambda x: x['logit'])[:3]
    categories = [
        ("Realest", realest),
        ("Most Uncertain", uncertain),
        ("Best Fakes", fooled),
        ("Worst Fakes", obvious),
    ]
    
    plot_vars = [cfg.selected_variable]
    if cfg.selected_variable == 'temperature' and 'u_component_of_wind' in cfg.variables and 'v_component_of_wind' in cfg.variables:
        plot_vars.append('wind_speed')

    for var_name in plot_vars:
        config = PLOT_CONFIGS.get(var_name, {'cmap': 'viridis', 'title': var_name, 'unit': ''})
        fig, axes = plt.subplots(4, 3, figsize=(18, 15), subplot_kw={'projection': ccrs.PlateCarree()})
        fig.suptitle(f"Discriminator Evaluation: {config['title']}", fontsize=20, y=0.98)

        for row_idx, (_, cat_list) in enumerate(categories):
            for col_idx, item in enumerate(cat_list):
                ax = axes[row_idx, col_idx]
                global_idx, logit = item['idx'], item['logit']
                metadata = sample_metadata(test_dataset, global_idx)

                map_slice = metadata["ds"].sel(time=metadata["time"])
                if metadata["lead_idx"] is not None:
                    map_slice = map_slice.isel(prediction_timedelta=metadata["lead_idx"])
                
                if var_name == 'wind_speed':
                    plot_data = (map_slice['u_component_of_wind']**2 + map_slice['v_component_of_wind']**2)**0.5
                else:
                    plot_data = map_slice[var_name]
                    
                plot_data.plot(ax=ax, x="longitude", y="latitude", transform=ccrs.PlateCarree(), cmap=config['cmap'], robust=True, add_colorbar=False)
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, alpha=0.5)
                ax.set_title(f"{metadata['data_type']} | Logit: {logit:.2f}\n{str(metadata['time'])[:13]}", fontsize=10)

        output_path = os.path.join(cfg.output_dir, f"evaluation_{cfg.model_name}_{var_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved evaluation plot: {output_path}")

if __name__ == "__main__":
    evaluate_and_visualize()
