import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Discriminator.scripts.manage_pipeline_storage import clean_wandb, inventory, migrate


class PipelineStorageTests(unittest.TestCase):
    def test_inventory_reports_run_status_and_bytes_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            run = home / "project" / "pipeline_runs" / "run-01"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text('{"status": "completed"}')
            (run / "payload.bin").write_bytes(b"12345")

            report = inventory(home)

            self.assertEqual(len(report["pipeline_runs"]), 1)
            self.assertEqual(report["pipeline_runs"][0]["status"], "completed")
            self.assertGreaterEqual(report["pipeline_runs"][0]["bytes"], 5)
            self.assertTrue(run.is_dir())

    def test_migration_is_dry_run_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "run-01"
            destination = root / "destination"
            source.mkdir(parents=True)
            (source / "result.txt").write_text("result")

            report = migrate(source, destination)

            self.assertFalse(report["apply"])
            self.assertFalse((destination / source.name).exists())
            self.assertTrue(source.exists())

    def test_applied_migration_verifies_and_optionally_removes_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "run-01"
            destination = root / "destination"
            source.mkdir(parents=True)
            (source / "nested").mkdir()
            primary = source / "nested" / "result.txt"
            primary.write_text("result")
            (source / "nested" / "result_notitle.txt").hardlink_to(primary)

            report = migrate(source, destination, apply=True, delete_source=True)

            self.assertEqual(report["verified_files"], 2)
            self.assertFalse(source.exists())
            migrated = destination / "run-01" / "nested"
            self.assertEqual((migrated / "result.txt").read_text(), "result")
            self.assertEqual(
                (migrated / "result.txt").stat().st_ino,
                (migrated / "result_notitle.txt").stat().st_ino,
            )

    def test_legacy_wandb_cleanup_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            legacy = home / "project" / "wandb"
            legacy.mkdir(parents=True)
            (legacy / "payload.bin").write_bytes(b"123")
            report = clean_wandb(home, legacy_roots=[legacy])
            self.assertEqual(report["legacy_roots"][0]["bytes"], 3)
            self.assertTrue(legacy.exists())
            with patch("Discriminator.scripts.manage_pipeline_storage._wandb_process_active", return_value=False):
                clean_wandb(home, apply=True, legacy_roots=[legacy])
            self.assertFalse(legacy.exists())
            with self.assertRaisesRegex(ValueError, "unsafe"):
                clean_wandb(home, legacy_roots=[home / "project"])


if __name__ == "__main__":
    unittest.main()
