"""Render the blind-spot figure and tables against the resampled null.

The shipped figure and table judge every corruption against one ERA5-vs-ERA5
null draw. `evaluate_bootstrap_null.py` replaces that with a distribution; this
script turns its CSVs into the two deliverables:

* the per-corruption severity figure, with the null shown as a median diamond
  and a p5-p95 whisker instead of a bare point, and each corruption curve drawn
  as the mean of its paired resamples with a +/-1 sd ribbon;
* the blind-spot table (N and M), plus a companion table giving the raw
  exceedance count -- how many of the null draws each score actually beats --
  so the pass/fail threshold is explicit rather than implied by a single number.

Reading from CSV rather than recomputing keeps this cheap: the band definition,
the styling and the table formatting can all be changed locally in seconds
without re-running the cluster job. Point it at a directory with
`bootstrap_null.data_dir=<path>` to replot results copied off the cluster.
"""

import csv
import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

try:
    from . import plot_standard_metric_baselines as P
    from .plot_bundles import save_figure_bundle
    from .evaluate_bootstrap_null import bootstrap_get, displayed
except ImportError:
    import plot_standard_metric_baselines as P
    from plot_bundles import save_figure_bundle
    from evaluate_bootstrap_null import bootstrap_get, displayed


# Column identity, established by reproducing the published table against the
# shipped single null: these four choices match it 9/9, 9/9, 8/9 and 9/9.
TABLE_COLUMNS = [
    ("SCWD", "scwd"),
    ("Mean WD", "global_mean_wasserstein"),
    ("Zonal spec.", "zonal_energy_spectrum_log_l2"),
    ("Raw MMD", "mmd_rbf"),
]

# Row order and grouping of the published table.
TABLE_ROWS = [
    "hemisphere_splice",
    "checkerboard_2px",
    "equatorial_checker_texture",
    "zonal_scanlines",
    "meridional_scanlines",
    None,  # \midrule
    "gaussian_blur",
    "grf",
    "hf_noise",
    "pixel_replace",
]

# The learned metric is not produced by this sweep -- it comes from the trained
# discriminator, which has its own null. Carried over verbatim from the
# published table and flagged as such, NOT re-derived against the resampled
# threshold.
CARRIED_OVER_OURS = {name: (True, True) for name in TABLE_ROWS if name}

# The one cell where reproducing the published table disagrees with a rerun.
KNOWN_DISCREPANCY = ("grf", "zonal_energy_spectrum_log_l2")


def table_layout(verdicts):
    """Published row order, restricted to what this run actually scored.

    A partial run (a smoke test, or a config with fewer corruptions) should
    still render. Anything scored but absent from the published layout is
    appended rather than dropped silently, and both differences are reported.
    """
    scored = [row["corruption"] for row in verdicts]
    layout, missing = [], []
    for corruption in TABLE_ROWS:
        if corruption is None:
            layout.append(None)
        elif corruption in scored:
            layout.append(corruption)
        else:
            missing.append(corruption)
    extra = [name for name in scored if name not in TABLE_ROWS]
    if extra:
        layout.extend([None, *extra])
    while layout and layout[-1] is None:
        layout.pop()
    return layout, missing, extra


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_outputs(data_root):
    """Load everything `evaluate_bootstrap_null` wrote."""
    summary = json.loads((data_root / "bootstrap_null_summary.json").read_text())
    curves = read_rows(data_root / "bootstrap_curves.csv")
    draws = read_rows(data_root / "bootstrap_null_draws.csv")
    verdicts = read_rows(data_root / "bootstrap_null_verdicts.csv")
    return summary, curves, draws, verdicts


def curve_table(curves, metrics):
    """Index resampled scores by (corruption, severity), in display convention."""
    grouped = {}
    for row in curves:
        key = (row["corruption"], float(row["severity"]))
        grouped.setdefault(key, []).append(row)
    table = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["replicate"]))
        table[key] = {
            name: displayed(name, [float(r[name]) for r in rows]) for name in metrics
        }
    return table


def null_table(draws, metrics, scheme="day_partition"):
    """Null draws under one sampling scheme, in display convention.

    `blindspots_from_temporal_draws` writes the same schema under its own
    scheme name, so the summary carries the prefix rather than assuming it.
    """
    return {
        name: displayed(name, [float(r[f"{scheme}__{name}"]) for r in draws])
        for name in metrics
    }


def severity_ladder(table, corruption):
    return sorted({severity for (name, severity) in table if name == corruption})


def render_figure(table, nulls, summary, output_path, variable):
    """One panel per corruption, resampled ribbons, null as median + whisker."""
    metrics = P.plotted_metric_names(summary["metrics"])
    if not metrics:
        metrics = list(summary["metrics"])
    corruptions = sorted({name for (name, _) in table})
    quantile = float(summary["threshold_quantile"])

    # Same "metric maximum = 1" convention as the shipped figure. Values handed
    # to the helper are already non-negative, so its abs() is a no-op and the
    # scales match what is actually drawn.
    scale_rows = [
        {name: float(values[name].mean()) for name in metrics}
        for values in table.values()
    ] + [
        {name: float(np.quantile(nulls[name], quantile)) for name in metrics}
    ]
    scales = P.metric_normalization_scales(scale_rows, metrics)
    colors = P.metric_colors(metrics)

    n_cols = 2
    n_rows = int(np.ceil(len(corruptions) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.8 * n_rows), squeeze=False)
    for axis, corruption in zip(axes.ravel(), corruptions):
        severities = severity_ladder(table, corruption)
        for name in metrics:
            scale = scales[name]
            mean = np.array([table[(corruption, s)][name].mean() for s in severities]) / scale
            sd = np.array([
                table[(corruption, s)][name].std(ddof=1)
                if table[(corruption, s)][name].size > 1 else 0.0
                for s in severities
            ]) / scale
            axis.plot(severities, mean, marker="o", linewidth=1.6,
                      color=colors[name], label=name)
            axis.fill_between(severities, mean - sd, mean + sd,
                              color=colors[name], alpha=0.22, linewidth=0)
            draws = nulls[name] / scale
            median = float(np.median(draws))
            axis.errorbar(
                [0.0], [median],
                yerr=[[median - float(np.quantile(draws, 1.0 - quantile))],
                      [float(np.quantile(draws, quantile)) - median]],
                fmt="D", markersize=5, color=colors[name], ecolor=colors[name],
                elinewidth=1.2, capsize=2.5, zorder=4,
            )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set(title=corruption, xlabel="Corruption severity")
        axis.grid(True, alpha=0.3)
    for axis in axes.ravel()[len(corruptions):]:
        axis.axis("off")

    figure.supylabel("Normalized divergence (metric maximum = 1)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure.suptitle(
        f"Normalized Distributional Metrics by Corruption: {variable}\n"
        f"Ribbons: +/-1 sd over {summary['curve_replicates']} paired resamples.  "
        f"Diamonds at 0: resampled ERA5 null, median with p{(1-quantile)*100:g}-p{quantile*100:g} whisker.",
        fontsize=13,
    )
    figure.tight_layout(rect=[0.03, 0, 0.82, 0.90])

    payload = payload_arrays(table, nulls, summary, scales, metrics, corruptions)
    paths = save_figure_bundle(
        figure, output_path, plot_type="bootstrap_blindspots", payload=payload,
        metadata={"threshold_quantile": quantile,
                  "curve_replicates": int(summary["curve_replicates"]),
                  "null_replicates": int(summary["replicates"])},
        dpi=220, bbox_inches="tight",
    )
    plt.close(figure)
    return paths


def payload_arrays(table, nulls, summary, scales, metrics, corruptions):
    """Raw per-replicate scores, so any band definition can be recomputed later.

    `save_figure_bundle` captures the rendered artists, which would only give
    back the mean line and the +/-1 sd edges actually drawn. Shipping the raw
    resamples means p5-p95, IQR or 2 sd can be substituted from the NPZ alone,
    with no cluster re-run.
    """
    ladders = [severity_ladder(table, corruption) for corruption in corruptions]
    n_severity = max(len(ladder) for ladder in ladders)
    n_replicate = max(values[metrics[0]].size for values in table.values())

    severities = np.full((len(corruptions), n_severity), np.nan)
    scores = np.full((len(corruptions), n_severity, n_replicate, len(metrics)), np.nan)
    for c_index, (corruption, ladder) in enumerate(zip(corruptions, ladders)):
        for s_index, severity in enumerate(ladder):
            severities[c_index, s_index] = severity
            for m_index, name in enumerate(metrics):
                values = table[(corruption, severity)][name]
                scores[c_index, s_index, : values.size, m_index] = values
    return {
        "curve_scores": scores,
        "curve_severities": severities,
        "null_draws": np.stack([nulls[name] for name in metrics], axis=1),
        "corruptions": np.asarray(corruptions, dtype="<U40"),
        "metrics": np.asarray(metrics, dtype="<U40"),
        "normalization_scales": np.asarray([scales[name] for name in metrics]),
        "null_thresholds": np.asarray([
            float(summary["thresholds"][name]) for name in metrics
        ]),
    }


def mark(flag):
    return "$\\checkmark$" if flag else "$\\times$"


def blindspot_tex(verdicts, summary):
    """Regenerate the paper table with N against the resampled threshold."""
    by_corruption = {row["corruption"]: row for row in verdicts}
    layout, _, _ = table_layout(verdicts)
    columns = [(label, metric) for label, metric in TABLE_COLUMNS if metric in summary["metrics"]]
    quantile = float(summary["threshold_quantile"])
    lines = [
        "% Regenerated by scripts/plot_bootstrap_blindspots.py.",
        f"% N: score at maximum severity exceeds the p{quantile*100:g} of a "
        f"{summary['replicates']}-draw resampled ERA5 null.",
        "% M: score is non-decreasing across every non-zero severity step.",
        "% The 'Ours' column is CARRIED OVER from the published table and was NOT",
        "% re-derived against the resampled threshold -- it still rests on the old",
        "% single null.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{",
        "  Response of each metric to controlled corruptions.",
        f"  \\textbf{{N}}: score exceeds the resampled null threshold (p{quantile*100:g} of "
        f"{summary['replicates']} draws) at the highest tested severity;",
        "  \\textbf{M}: score increases monotonically with corruption severity.",
        "  }",
        "  \\label{tab:blindspots}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\begin{tabular}{@{}l |" + "cc|" * (len(columns) + 1) + " @{} }",
        "    \\toprule",
    ]
    headers = [label for label, _ in columns] + ["Ours"]
    lines.append("    & " + "\n    & ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in headers) + " \\\\")
    for index in range(len(headers)):
        closer = "l" if index == len(headers) - 1 else "lr"
        lines.append(f"    \\cmidrule({closer}){{{2 + 2 * index}-{3 + 2 * index}}}")
    lines.append("    Corruption")
    lines.extend("    & N & M" for _ in headers)
    lines[-1] += " \\\\"
    lines.append("    \\midrule")

    for corruption in layout:
        if corruption is None:
            lines.append("")
            lines.append("    \\midrule")
            continue
        row = by_corruption[corruption]
        cells = []
        for _, metric in columns:
            cells.append((row[f"{metric}__detected"] == "True",
                          row[f"{metric}__monotone"] == "True"))
        cells.append(CARRIED_OVER_OURS.get(corruption, (False, False)))
        escaped = corruption.replace("_", "\\_")
        lines.append("")
        lines.append(f"    \\texttt{{{escaped}}}")
        for index, (n_flag, m_flag) in enumerate(cells):
            terminator = " \\\\" if index == len(cells) - 1 else ""
            lines.append(f"      & {mark(n_flag)} & {mark(m_flag)}{terminator}")
    lines.extend(["", "    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def exceedance_tex(verdicts, summary):
    """How many null draws each maximum-severity score actually beats."""
    metrics = summary["metrics"]
    by_corruption = {row["corruption"]: row for row in verdicts}
    layout, _, _ = table_layout(verdicts)
    replicates = int(summary["replicates"])
    short = {name: name.replace("_", " ").replace("energy spectrum", "spec")[:14] for name in metrics}
    lines = [
        "% Regenerated by scripts/plot_bootstrap_blindspots.py.",
        f"% Each cell counts how many of the {replicates} resampled ERA5 null draws the",
        "% maximum-severity score exceeds. A detection needs to clear the p"
        f"{float(summary['threshold_quantile'])*100:g}.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        f"  \\caption{{Null draws exceeded at maximum severity, out of {replicates}.}}",
        "  \\label{tab:null-exceedance}",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{@{}l" + "r" * len(metrics) + "@{}}",
        "    \\toprule",
        "    Corruption & " + " & ".join(
            f"\\rotatebox{{60}}{{\\texttt{{{short[name]}}}}}" for name in metrics
        ) + " \\\\",
        "    \\midrule",
    ]
    for corruption in layout:
        if corruption is None:
            lines.append("    \\midrule")
            continue
        row = by_corruption[corruption]
        cells = [f"{int(row[f'{name}__null_draws_exceeded'])}" for name in metrics]
        escaped = corruption.replace("_", "\\_")
        lines.append(f"    \\texttt{{{escaped}}} & " + " & ".join(cells) + " \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def exceedance_csv(verdicts, summary, path):
    metrics = summary["metrics"]
    fieldnames = ["corruption", "metric", "score", "threshold", "margin",
                  "null_draws_exceeded", "null_replicates", "p_value",
                  "detected", "monotone", "monotone_fraction"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in verdicts:
            for name in metrics:
                writer.writerow({
                    "corruption": row["corruption"], "metric": name,
                    "score": row[f"{name}__value"], "threshold": row[f"{name}__threshold"],
                    "margin": row[f"{name}__margin"],
                    "null_draws_exceeded": row[f"{name}__null_draws_exceeded"],
                    "null_replicates": summary["replicates"],
                    "p_value": row[f"{name}__p_value"],
                    "detected": row[f"{name}__detected"], "monotone": row[f"{name}__monotone"],
                    "monotone_fraction": row[f"{name}__monotone_fraction"],
                })


def plot_bootstrap_blindspots(cfg):
    """Render every bootstrap-null deliverable; returns the rendered PNG paths."""
    variables = P.variables_from_config(cfg)
    configured = bootstrap_get(cfg, "data_dir", None)
    output_root = (
        Path(str(configured)).parent if configured
        else P.baseline_output_dir(cfg, variables) / "bootstrap_null"
    )
    data_root = Path(str(configured)) if configured else output_root / "data"

    summary, curves, draws, verdicts = load_outputs(data_root)
    metrics = summary["metrics"]
    table = curve_table(curves, metrics)
    nulls = null_table(draws, metrics, summary.get("null_scheme", "day_partition"))

    figure_path = output_root / "plots" / "corruption_by_type_bootstrap_null.png"
    paths = render_figure(table, nulls, summary, figure_path, P.joint_variable_name(variables))

    (data_root / "blindspot_table.tex").write_text(blindspot_tex(verdicts, summary))
    (data_root / "null_exceedance_table.tex").write_text(exceedance_tex(verdicts, summary))
    exceedance_csv(verdicts, summary, data_root / "null_exceedance.csv")

    report(verdicts, summary)
    print(f"\nWrote {figure_path} (+ .pdf/.npz) and tables in {data_root}")
    return [figure_path]


def report(verdicts, summary):
    """Print the verdict grid and name any cell that needs a human decision."""
    metrics = summary["metrics"]
    replicates = int(summary["replicates"])
    print(f"\n{'corruption':30s} " + "".join(f"{name[:16]:>19s}" for name in metrics))
    for row in verdicts:
        cells = []
        for name in metrics:
            n_flag = "N" if row[f"{name}__detected"] == "True" else "-"
            m_flag = "M" if row[f"{name}__monotone"] == "True" else "-"
            beat = int(row[f"{name}__null_draws_exceeded"])
            cells.append(f"{n_flag}{m_flag} {beat:3d}/{replicates}".rjust(19))
        print(f"{row['corruption']:30s} " + "".join(cells))

    _, missing, extra = table_layout(verdicts)
    if missing:
        print(f"\nNot scored in this run, omitted from the tables: {', '.join(missing)}")
    if extra:
        print(f"Scored but absent from the published layout, appended: {', '.join(extra)}")

    corruption, metric = KNOWN_DISCREPANCY
    row = next((r for r in verdicts if r["corruption"] == corruption), None)
    if row is not None:
        print(
            f"\nCheck by hand: the published table marks {corruption} x Zonal spec. as a miss, "
            f"but this run scores it {float(row[f'{metric}__margin']):.2f}x the threshold "
            f"({int(row[f'{metric}__null_draws_exceeded'])}/{replicates} null draws beaten). "
            "Re-derive that cell before publishing."
        )
    print("The 'Ours' column in blindspot_table.tex is carried over, not re-derived.")
    if summary.get("null_definition"):
        print(f"Null: {summary['null_definition']}.")


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config")
def main(cfg: DictConfig):
    plot_bootstrap_blindspots(cfg)


if __name__ == "__main__":
    main()
