"""Portable figure bundles and centralized dashboard/paper rendering."""

import fnmatch
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl

import numpy as np


_DEFAULT_SAVE_PDF = False
_PLOT_PROFILE = "dashboard"
_PAPER_WIDTH_INCHES = 5.5
_PAPER_WIDTH_KIND = "full"
_PAPER_COLUMN_GAP_INCHES = 0.12
_INCLUDE_PATTERNS = ()


# Shared categorical palette for dashboard and manuscript plots. Model/metric
# series reuse the four non-critic colors and remain distinguishable through markers.
PLOT_PALETTE = (
    "#AD99FF",  # Soft Periwinkle
    "#030027",  # Prussian Blue
    "#5CA4A9",  # Tropical Teal
    "#DC9E82",  # Light Bronze
    "#AB494B",  # Dusty Mauve
)
REFERENCE_COLOR = PLOT_PALETTE[0]
CANDIDATE_COLOR = PLOT_PALETTE[-1]
# Reserve Prussian Blue for the learned critic throughout the paper. It is
# intentionally excluded from automatically assigned model/metric colors.
CRITIC_COLOR = PLOT_PALETTE[1]
SERIES_PALETTE = (
    PLOT_PALETTE[0], PLOT_PALETTE[2], PLOT_PALETTE[3], PLOT_PALETTE[4],
)
# Six distinct non-critic colors for the standard 6/12/24/48/96/192 h set.
# Start from teal so the first outline does not duplicate the ERA5 fill.
LEAD_TIME_PALETTE = (
    PLOT_PALETTE[2], PLOT_PALETTE[3], PLOT_PALETTE[4],
    "#C58F2A",  # Muted ochre
    "#356859",  # Deep pine
    PLOT_PALETTE[0],
)


MODEL_COLOR_OVERRIDES = {
    "era5 forecast": "#030027",  # Prussian/midnight blue
    "ucast member 0": "#C58F2A",  # Muted ochre
    "ucast": "#C58F2A",
    "swift": "#356859",  # Deep pine
}

MODEL_DISPLAY_NAMES = {
    "ucast member 0": "UCast",
    "ucast": "UCast",
}


def normalized_model_name(label):
    """Canonicalize a model identifier for display/color lookup only."""
    return " ".join(str(label).replace("_", " ").split()).lower()


def displayed_model_name(label):
    """Return a concise plot label without changing persisted target IDs."""
    return MODEL_DISPLAY_NAMES.get(normalized_model_name(label), str(label))


def categorical_colors(count, offset=0):
    """Return non-critic series colors, reserving Prussian Blue for ours."""
    return [
        SERIES_PALETTE[(int(offset) + index) % len(SERIES_PALETTE)]
        for index in range(int(count))
    ]


def lead_time_colors(count):
    """Return distinct ordered colors for multi-lead overlays."""
    return [LEAD_TIME_PALETTE[index % len(LEAD_TIME_PALETTE)] for index in range(int(count))]


def model_colors(labels, offset=0):
    """Return stable model colors, including overrides for selected forecasts."""
    labels = [str(label) for label in labels]
    defaults = categorical_colors(len(labels), offset=offset)
    colors = []
    for label, default in zip(labels, defaults):
        normalized = normalized_model_name(label)
        colors.append(MODEL_COLOR_OVERRIDES.get(normalized, default))
    return colors


def configure_plot_bundle_saving(*, save_pdf=False, profile="dashboard",
                                 paper_width_inches=5.5, paper_width_kind="full",
                                 paper_column_gap_inches=0.12, include_patterns=None):
    """Configure process-wide output behavior used by figure bundle writers."""
    global _DEFAULT_SAVE_PDF, _PLOT_PROFILE, _PAPER_WIDTH_INCHES
    global _PAPER_WIDTH_KIND, _PAPER_COLUMN_GAP_INCHES, _INCLUDE_PATTERNS
    profile = str(profile).lower()
    if profile not in {"dashboard", "paper"}:
        raise ValueError(f"Unknown plotting profile: {profile!r}")
    if paper_width_kind not in {"full", "half"}:
        raise ValueError("paper_width_kind must be 'full' or 'half'.")
    _PLOT_PROFILE = profile
    _DEFAULT_SAVE_PDF = bool(save_pdf)
    _PAPER_WIDTH_INCHES = float(paper_width_inches)
    _PAPER_WIDTH_KIND = str(paper_width_kind)
    _PAPER_COLUMN_GAP_INCHES = float(paper_column_gap_inches)
    _INCLUDE_PATTERNS = tuple(str(value) for value in (include_patterns or ()))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=SERIES_PALETTE)
    mpl.rcParams["font.family"] = ["Nimbus Roman", "serif"]
    mpl.rcParams["font.serif"] = [
        "Nimbus Roman", "DejaVu Serif",
    ]
    # STIX closely matches Times-style LaTeX mathematics; make math text italic
    # unless an expression explicitly requests another style.
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["mathtext.default"] = "it"


def configure_plot_bundle_saving_from_cfg(cfg):
    """Configure bundle output from the shared plotting config section."""
    settings = cfg.get("plotting", {}) or {}
    paper = settings.get("paper", {}) or {}
    configure_plot_bundle_saving(
        save_pdf=bool(settings.get("save_pdf", False)),
        profile=str(settings.get("profile", "dashboard")),
        paper_width_inches=float(paper.get("text_width_inches", 5.5)),
        paper_width_kind=str(paper.get("width", "full")),
        paper_column_gap_inches=float(paper.get("column_gap_inches", 0.12)),
        include_patterns=paper.get("include", []),
    )


def plot_bundle_paths(png_path):
    """Return the three side-by-side files belonging to a rendered figure."""
    png_path = Path(png_path)
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Plot bundle path must end in .png, got {png_path}")
    return png_path, png_path.with_suffix(".pdf"), png_path.with_suffix(".npz")


def profiled_plot_path(png_path):
    """Redirect paper output below the plot tree without touching dashboards."""
    path = Path(png_path)
    if _PLOT_PROFILE != "paper" or "paper" in path.parts:
        return path
    parts = list(path.parts)
    try:
        index = parts.index("plots")
    except ValueError:
        return path.with_name(f"{path.stem}_paper{path.suffix}")
    parts.insert(index + 1, "paper")
    return Path(*parts)


def _selected(path):
    if not _INCLUDE_PATTERNS:
        return True
    value = Path(path).as_posix()
    return any(
        fnmatch.fnmatch(value, pattern) or fnmatch.fnmatch(Path(value).name, pattern)
        for pattern in _INCLUDE_PATTERNS
    )


def _paper_width(width_kind=None):
    width_kind = _PAPER_WIDTH_KIND if width_kind is None else str(width_kind)
    if width_kind == "half":
        return (_PAPER_WIDTH_INCHES - _PAPER_COLUMN_GAP_INCHES) / 2.0
    if width_kind == "full":
        return _PAPER_WIDTH_INCHES
    raise ValueError("paper_width_kind must be 'full' or 'half'.")


def _paperize_figure(figure, paper_width_kind=None):
    """Resize and restyle an existing figure at its final manuscript width."""
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["mathtext.fontset"] = "stix"
    old_width, old_height = figure.get_size_inches()
    width = _paper_width(paper_width_kind)
    figure.set_size_inches(width, width * old_height / max(old_width, 1e-12), forward=True)
    for text_artist in figure.findobj(match=lambda artist: hasattr(artist, "set_fontsize")):
        try:
            text_artist.set_fontsize(7.0)
            text_artist.set_fontfamily("serif")
        except (TypeError, ValueError):
            pass
    if getattr(figure, "_suptitle", None) is not None:
        figure._suptitle.set_fontsize(8.0)
    for axis in figure.axes:
        axis.title.set_fontsize(8.0)
        axis.xaxis.label.set_fontsize(8.0)
        axis.yaxis.label.set_fontsize(8.0)
        axis.tick_params(labelsize=7.0)
        legend = axis.get_legend()
        if legend is not None:
            for text_artist in legend.get_texts():
                text_artist.set_fontsize(6.5)
        for line in axis.lines:
            line.set_linewidth(min(float(line.get_linewidth()), 1.1))
            if line.get_marker() not in {None, "None", ""}:
                line.set_markersize(min(float(line.get_markersize()), 3.5))
    for legend in figure.legends:
        # The creating plot owns legend placement and reserves its surrounding
        # margin. Paperization changes typography, not layout semantics.
        for text_artist in legend.get_texts():
            text_artist.set_fontsize(6.5)


def titleless_plot_path(png_path):
    """Return the title-less companion path for a rendered PNG."""
    png_path = Path(png_path)
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Plot bundle path must end in .png, got {png_path}")
    return png_path.with_name(f"{png_path.stem}_notitle.png")


def all_plot_bundle_paths(png_path):
    """Return the normal and title-less PNG/PDF/NPZ bundles for a figure."""
    png_path = profiled_plot_path(png_path)
    return (*plot_bundle_paths(png_path), *plot_bundle_paths(titleless_plot_path(png_path)))


def _safe_array(value):
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("Plot-bundle arrays must not use object dtype.")
    return array


def _artist_arrays(figure):
    """Capture the numerical artists actually rendered by Matplotlib."""
    arrays = {}
    metadata_axes = []
    for axis_index, axis in enumerate(figure.axes):
        axis_metadata = {
            "title": axis.get_title(), "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
            "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
            "xlim": list(map(float, axis.get_xlim())), "ylim": list(map(float, axis.get_ylim())),
            "position": list(map(float, axis.get_position().bounds)),
            "projection": type(axis).__name__, "lines": [], "collections": [],
            "images": [], "patches": [],
        }
        metadata_axes.append(axis_metadata)
        for line_index, line in enumerate(axis.lines):
            prefix = f"axes_{axis_index}_line_{line_index}"
            arrays[f"{prefix}_x"] = _safe_array(line.get_xdata(orig=False))
            arrays[f"{prefix}_y"] = _safe_array(line.get_ydata(orig=False))
            axis_metadata["lines"].append({
                "prefix": prefix, "label": line.get_label(),
                "color": mpl.colors.to_hex(line.get_color(), keep_alpha=True),
                "linestyle": line.get_linestyle(), "marker": str(line.get_marker()),
                "linewidth": float(line.get_linewidth()), "markersize": float(line.get_markersize()),
            })
        for collection_index, collection in enumerate(axis.collections):
            prefix = f"axes_{axis_index}_collection_{collection_index}"
            collection_metadata = {
                "prefix": prefix, "label": collection.get_label(),
                "type": type(collection).__name__,
            }
            if getattr(collection, "cmap", None) is not None:
                collection_metadata["cmap"] = collection.cmap.name
            norm = getattr(collection, "norm", None)
            if norm is not None:
                collection_metadata["vmin"] = None if norm.vmin is None else float(norm.vmin)
                collection_metadata["vmax"] = None if norm.vmax is None else float(norm.vmax)
            axis_metadata["collections"].append(collection_metadata)
            if hasattr(collection, "get_offsets"):
                offsets = collection.get_offsets()
                if offsets is not None and np.asarray(offsets).size:
                    arrays[f"{prefix}_offsets"] = _safe_array(offsets)
            if hasattr(collection, "get_array"):
                values = collection.get_array()
                if values is not None:
                    arrays[f"{prefix}_values"] = _safe_array(values)
            if hasattr(collection, "get_coordinates"):
                coordinates = collection.get_coordinates()
                if coordinates is not None:
                    arrays[f"{prefix}_coordinates"] = _safe_array(coordinates)
        for image_index, image in enumerate(axis.images):
            prefix = f"axes_{axis_index}_image_{image_index}"
            arrays[prefix] = _safe_array(image.get_array())
            arrays[f"{prefix}_extent"] = _safe_array(image.get_extent())
            axis_metadata["images"].append({
                "prefix": prefix, "cmap": image.get_cmap().name,
                "vmin": None if image.norm.vmin is None else float(image.norm.vmin),
                "vmax": None if image.norm.vmax is None else float(image.norm.vmax),
                "origin": str(image.origin),
            })
        for patch_index, patch in enumerate(axis.patches):
            prefix = f"axes_{axis_index}_patch_{patch_index}"
            if all(hasattr(patch, method) for method in ("get_x", "get_y", "get_width", "get_height")):
                geometry = "rectangle"
                arrays[prefix] = _safe_array([
                    patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height(),
                ])
            elif hasattr(patch, "get_path") and hasattr(patch, "get_patch_transform"):
                geometry = "path"
                vertices = patch.get_patch_transform().transform(patch.get_path().vertices)
                arrays[f"{prefix}_vertices"] = _safe_array(vertices)
            else:
                continue
            axis_metadata["patches"].append({
                "prefix": prefix, "geometry": geometry, "label": patch.get_label(),
                "facecolor": mpl.colors.to_hex(patch.get_facecolor(), keep_alpha=True),
                "edgecolor": mpl.colors.to_hex(patch.get_edgecolor(), keep_alpha=True),
                "linewidth": float(patch.get_linewidth()),
                "alpha": patch.get_alpha(),
            })
    return arrays, metadata_axes


@contextmanager
def without_suptitle(figure):
    """Temporarily remove the main title used by the title-less companion.

    Most figures use ``suptitle``.  Single-panel figures often place their
    effective figure title on the sole plotting axis instead; remove that too,
    while leaving multiple panel titles intact as contextual labels.
    """
    supertitle = getattr(figure, "_suptitle", None)
    supertitle_text = None if supertitle is None else supertitle.get_text()
    titled_axes = []
    for axis in figure.get_axes():
        if not axis.get_visible():
            continue
        titles = [(location, axis.get_title(loc=location))
                  for location in ("left", "center", "right")]
        titles = [(location, value) for location, value in titles if value]
        if titles:
            titled_axes.append((axis, titles))
    axis_titles = titled_axes if len(titled_axes) == 1 else []
    try:
        if supertitle is not None:
            supertitle.set_text("")
        for axis, titles in axis_titles:
            for location, _ in titles:
                axis.set_title("", loc=location)
        yield
    finally:
        if supertitle is not None:
            supertitle.set_text(supertitle_text)
        for axis, titles in axis_titles:
            for location, value in titles:
                axis.set_title(value, loc=location)


def rasterize_field_artists(figure):
    """Rasterize dense field artists in PDF while retaining vector annotations."""
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage

    count = 0
    for axis in figure.get_axes():
        for artist in list(axis.collections) + list(axis.images):
            if isinstance(artist, (Collection, AxesImage)) and not artist.get_rasterized():
                artist.set_rasterized(True)
                count += 1
    return count


def _atomic_figure_save(figure, path, **kwargs):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}", dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, **kwargs)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path, payload):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _save_one_figure_bundle(figure, png_path, *, plot_type, payload, metadata, dpi,
                            bbox_inches, capture_artists=True, save_pdf=False, shared_npz=None):
    png_path, pdf_path, npz_path = plot_bundle_paths(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": int(dpi)}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    _atomic_figure_save(figure, png_path, **save_kwargs)
    if save_pdf:
        rasterize_field_artists(figure)
        _atomic_figure_save(figure, pdf_path, dpi=int(dpi), bbox_inches=bbox_inches)

    if shared_npz is None:
        arrays, axes = _artist_arrays(figure) if capture_artists else ({}, [])
        for key, value in (payload or {}).items():
            arrays[f"input_{key}"] = _safe_array(value)
        supertitle = getattr(figure, "_suptitle", None)
        bundle_metadata = {
            "schema_version": 2,
            "figure_size_inches": list(map(float, figure.get_size_inches())),
            "plot_type": str(plot_type),
            "png": png_path.name,
            "pdf": pdf_path.name if save_pdf else None,
            "npz": npz_path.name,
            "axes": axes,
            "figure": {
                "suptitle": None if supertitle is None else supertitle.get_text(),
            },
            **(metadata or {}),
        }
        arrays["metadata_json"] = np.asarray(json.dumps(bundle_metadata, sort_keys=True))
        temporary_npz = npz_path.with_name(f".{npz_path.stem}.tmp.npz")
        try:
            np.savez_compressed(temporary_npz, **arrays)
            temporary_npz.replace(npz_path)
        except BaseException:
            temporary_npz.unlink(missing_ok=True)
            raise
    else:
        shared_npz = Path(shared_npz)
        npz_path.unlink(missing_ok=True)
        try:
            npz_path.hardlink_to(shared_npz)
        except OSError:
            npz_path.symlink_to(shared_npz.name)

    manifest_path = png_path.parent / "plot_data_manifest.json"
    manifest = {"schema_version": 2, "plots": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            pass
    manifest.setdefault("plots", {})[png_path.stem] = {
        "plot_type": str(plot_type), "png": png_path.name,
        "pdf": pdf_path.name if save_pdf else None, "npz": npz_path.name,
        "shared_npz": None if shared_npz is None else Path(shared_npz).name,
        **({"titleless": True} if shared_npz is not None else {}),
    }
    _atomic_json(manifest_path, manifest)
    return [png_path, *([pdf_path] if save_pdf else []), npz_path]


def save_figure_bundle(figure, png_path, *, plot_type, payload=None, metadata=None,
                       dpi=220, bbox_inches=None, capture_artists=True, save_pdf=None,
                       paper_width_kind=None, paperize=True):
    """Save titled and title-less PNG/NPZ bundles, optionally including PDFs.

    Large composite galleries may set ``paperize=False`` when their layout is
    already authored at its final aspect and typography; resizing those figures
    after layout changes the relative text/colorbar geometry.
    """
    png_path = profiled_plot_path(png_path)
    if not _selected(png_path):
        return []
    if _PLOT_PROFILE == "paper" and paperize:
        _paperize_figure(figure, paper_width_kind=paper_width_kind)
        bbox_inches = None
    if save_pdf is None:
        save_pdf = _DEFAULT_SAVE_PDF
    paths = _save_one_figure_bundle(
        figure, png_path, plot_type=plot_type, payload=payload, metadata=metadata,
        dpi=dpi, bbox_inches=bbox_inches, capture_artists=capture_artists, save_pdf=save_pdf,
    )
    with without_suptitle(figure):
        paths.extend(_save_one_figure_bundle(
            figure, titleless_plot_path(png_path), plot_type=plot_type,
            payload=payload, metadata={**(metadata or {}), "titleless": True},
            dpi=dpi, bbox_inches=bbox_inches, capture_artists=False, save_pdf=save_pdf,
            shared_npz=plot_bundle_paths(png_path)[2],
        ))
    return paths
