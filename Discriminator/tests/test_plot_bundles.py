import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Discriminator.scripts.plot_bundles import (
    all_plot_bundle_paths, configure_plot_bundle_saving, displayed_model_name, lead_time_colors, model_colors, rasterize_field_artists,
    save_figure_bundle, titleless_plot_path, without_suptitle,
)
from Discriminator.scripts.render_npz_paper_plots import load_bundle, reconstruct


class PlotBundleTests(unittest.TestCase):
    def test_ucast_member_zero_has_concise_display_name(self):
        self.assertEqual(displayed_model_name("UCast member 0"), "UCast")
        self.assertEqual(displayed_model_name("UCast_member_0"), "UCast")
        self.assertEqual(displayed_model_name("GraphCast"), "GraphCast")

    def test_standard_lead_times_use_six_distinct_extended_colors(self):
        colors = lead_time_colors(6)
        self.assertEqual(colors, [
            "#5CA4A9", "#DC9E82", "#AB494B",
            "#C58F2A", "#356859", "#AD99FF",
        ])
        self.assertEqual(len(set(colors)), 6)

    def test_selected_forecast_models_have_stable_distinct_colors(self):
        labels = ["GraphCast", "ERA5 Forecast", "UCast member 0", "SWIFT"]
        colors = model_colors(labels)
        self.assertEqual(colors[1:], ["#030027", "#C58F2A", "#356859"])
        self.assertEqual(
            model_colors(["SWIFT", "ERA5_Forecast", "UCast_member_0"]),
            ["#356859", "#030027", "#C58F2A"],
        )
        self.assertEqual(len(set(colors)), len(colors))

    def test_bundle_writes_png_and_safe_npz_sidecar_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "curve.png"
            figure, axis = plt.subplots()
            axis.plot([0.0, 1.0], [2.0, 3.0])
            axis.set_title("Panel title")
            figure.suptitle("Main title")
            paths = save_figure_bundle(
                figure, output, plot_type="unit_curve",
                payload={"raw_x": np.array([0.0, 1.0]), "raw_y": np.array([2.0, 3.0])},
            )
            plt.close(figure)
            self.assertEqual(paths, [
                output, output.with_suffix(".npz"),
                titleless_plot_path(output), titleless_plot_path(output).with_suffix(".npz"),
            ])
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertFalse(output.with_suffix(".pdf").exists())
            with np.load(output.with_suffix(".npz"), allow_pickle=False) as data:
                self.assertTrue(np.array_equal(data["input_raw_x"], [0.0, 1.0]))
                self.assertTrue(np.array_equal(data["input_raw_y"], [2.0, 3.0]))
                metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(metadata["plot_type"], "unit_curve")
            manifest = json.loads((output.parent / "plot_data_manifest.json").read_text())
            self.assertEqual(manifest["plots"]["curve"]["npz"], "curve.npz")
            self.assertEqual(manifest["plots"]["curve_notitle"]["png"], "curve_notitle.png")
            self.assertTrue(titleless_plot_path(output).is_file())
            titled_npz = output.with_suffix(".npz")
            titleless_npz = titleless_plot_path(output).with_suffix(".npz")
            self.assertEqual(titled_npz.stat().st_ino, titleless_npz.stat().st_ino)
            self.assertEqual(
                manifest["plots"]["curve_notitle"]["shared_npz"], "curve.npz",
            )
            self.assertTrue(manifest["plots"]["curve_notitle"]["titleless"])

    def test_titleless_context_removes_single_axis_main_title(self):
        figure, axis = plt.subplots()
        figure.suptitle("Figure heading")
        axis.set_title("Axis heading")
        try:
            with without_suptitle(figure):
                self.assertEqual(figure._suptitle.get_text(), "")
                self.assertEqual(axis.get_title(), "")
            self.assertEqual(figure._suptitle.get_text(), "Figure heading")
            self.assertEqual(axis.get_title(), "Axis heading")
        finally:
            plt.close(figure)

    def test_titleless_context_keeps_multi_panel_titles(self):
        figure, axes = plt.subplots(1, 2)
        figure.suptitle("Figure heading")
        axes[0].set_title("First panel")
        axes[1].set_title("Second panel")
        try:
            with without_suptitle(figure):
                self.assertEqual(figure._suptitle.get_text(), "")
                self.assertEqual(axes[0].get_title(), "First panel")
                self.assertEqual(axes[1].get_title(), "Second panel")
        finally:
            plt.close(figure)

    def test_bundle_writes_pdfs_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "curve.png"
            figure, axis = plt.subplots()
            axis.plot([0.0, 1.0], [2.0, 3.0])
            paths = save_figure_bundle(figure, output, plot_type="unit_curve", save_pdf=True)
            plt.close(figure)
            self.assertEqual(paths, list(all_plot_bundle_paths(output)))
            self.assertTrue(all(path.is_file() for path in paths))

    def test_bundle_rejects_pickle_requiring_payloads(self):
        figure, _ = plt.subplots()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                save_figure_bundle(
                    figure, Path(directory) / "bad.png", plot_type="bad",
                    payload={"objects": np.array([object()], dtype=object)},
                )
        plt.close(figure)

    def test_paper_profile_redirects_resizes_and_records_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run" / "plots" / "curve.png"
            figure, axis = plt.subplots(figsize=(10.0, 5.0))
            axis.set_xscale("log")
            axis.plot([1.0, 10.0], [2.0, 3.0], label="Metric", marker="s")
            configure_plot_bundle_saving(
                profile="paper", save_pdf=True, paper_width_inches=5.5, paper_width_kind="full"
            )
            try:
                paths = save_figure_bundle(figure, output, plot_type="unit_curve")
            finally:
                configure_plot_bundle_saving()
                plt.close(figure)
            paper = output.parent / "paper" / output.name
            self.assertEqual(paths[0], paper)
            self.assertTrue(paper.is_file())
            self.assertTrue(paper.with_suffix(".pdf").is_file())
            self.assertAlmostEqual(figure.get_size_inches()[0], 5.5)
            with np.load(paper.with_suffix(".npz"), allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
            self.assertEqual(metadata["schema_version"], 2)
            self.assertEqual(metadata["axes"][0]["xscale"], "log")
            self.assertEqual(metadata["axes"][0]["lines"][0]["label"], "Metric")
            self.assertEqual(metadata["axes"][0]["lines"][0]["marker"], "s")

    def test_paper_profile_can_preserve_authored_gallery_size(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run" / "plots" / "gallery.png"
            figure, _ = plt.subplots(figsize=(11.5, 14.0))
            configure_plot_bundle_saving(
                profile="paper", paper_width_inches=5.5, paper_width_kind="full"
            )
            try:
                paths = save_figure_bundle(
                    figure, output, plot_type="gallery", paperize=False,
                )
            finally:
                configure_plot_bundle_saving()
                plt.close(figure)
            self.assertTrue(paths[0].is_file())
            np.testing.assert_allclose(figure.get_size_inches(), [11.5, 14.0])

    def test_paper_profile_can_filter_filenames(self):
        configure_plot_bundle_saving(profile="paper", include_patterns=["keep*.png"])
        figure, _ = plt.subplots()
        try:
            self.assertEqual(
                save_figure_bundle(
                    figure, Path("/tmp/plots/drop.png"), plot_type="drop"
                ),
                [],
            )
        finally:
            configure_plot_bundle_saving()
            plt.close(figure)



    def test_schema_v2_bundle_can_reconstruct_named_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "source.png"
            figure, axis = plt.subplots(figsize=(6.0, 3.0))
            axis.plot([1.0, 2.0], [3.0, 5.0], label="Named line", color="tab:blue")
            axis.set(xlabel="x", ylabel="y", title="Panel")
            save_figure_bundle(figure, output, plot_type="unit_curve")
            plt.close(figure)
            archive, metadata = load_bundle(output.with_suffix(".npz"))
            try:
                rebuilt = reconstruct(archive, metadata, 5.5)
                self.assertEqual(len(rebuilt.axes), 1)
                self.assertEqual(rebuilt.axes[0].lines[0].get_label(), "Named line")
                np.testing.assert_allclose(rebuilt.axes[0].lines[0].get_xdata(), [1.0, 2.0])
                self.assertEqual(rebuilt.axes[0].get_title(), "Panel")
                plt.close(rebuilt)
            finally:
                archive.close()

    def test_schema_v2_bundle_reconstructs_filled_and_outline_histograms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "histogram.png"
            figure, axis = plt.subplots()
            axis.hist([0.0, 0.5, 1.0], bins=3, density=True, label="ERA5", alpha=0.4)
            axis.hist([0.2, 0.7, 0.9], bins=3, density=True, histtype="step", label="Forecast")
            source_patch_count = len(axis.patches)
            save_figure_bundle(figure, output, plot_type="unit_histogram")
            plt.close(figure)
            archive, metadata = load_bundle(output.with_suffix(".npz"))
            try:
                rebuilt = reconstruct(archive, metadata, 5.5)
                self.assertEqual(len(rebuilt.axes[0].patches), source_patch_count)
                labels = {patch.get_label() for patch in rebuilt.axes[0].patches}
                self.assertIn("ERA5", labels)
                self.assertIn("Forecast", labels)
                plt.close(rebuilt)
            finally:
                archive.close()

    def test_schema_v2_bundle_reconstructs_rasterized_field_map(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plots" / "map.png"
            figure, axis = plt.subplots()
            values = np.arange(6.0).reshape(2, 3)
            axis.pcolormesh(np.arange(4), np.arange(3), values, cmap="viridis")
            save_figure_bundle(figure, output, plot_type="unit_map")
            plt.close(figure)
            archive, metadata = load_bundle(output.with_suffix(".npz"))
            try:
                rebuilt = reconstruct(archive, metadata, 5.5)
                self.assertEqual(len(rebuilt.axes[0].collections), 1)
                np.testing.assert_allclose(
                    np.asarray(rebuilt.axes[0].collections[0].get_array()).reshape(values.shape),
                    values,
                )
                plt.close(rebuilt)
            finally:
                archive.close()


    def test_rasterizes_dense_field_artists_only_when_requested(self):
        figure, axis = plt.subplots()
        mesh = axis.pcolormesh(np.arange(4), np.arange(3), np.arange(6.0).reshape(2, 3))
        line, = axis.plot([0.0, 1.0], [0.0, 1.0])
        try:
            self.assertFalse(mesh.get_rasterized())
            self.assertEqual(rasterize_field_artists(figure), 1)
            self.assertTrue(mesh.get_rasterized())
            self.assertFalse(line.get_rasterized())
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
