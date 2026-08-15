from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tuner.spec import expand_path


class ExpandPathTests(unittest.TestCase):
    def test_preserves_huggingface_snapshot_symlink_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            blobs = root / "blobs"
            snapshot = root / "snapshots" / "abc123"
            blobs.mkdir()
            snapshot.mkdir(parents=True)

            blob = blobs / "deadbeef"
            blob.write_bytes(b"fake safetensors payload")

            model_link = snapshot / "model-00018-of-00018.safetensors"
            model_link.symlink_to(blob)

            expanded = expand_path(model_link)

            self.assertTrue(expanded.is_absolute())
            self.assertEqual(expanded.name, "model-00018-of-00018.safetensors")
            self.assertEqual(expanded.suffix, ".safetensors")
            self.assertTrue(expanded.is_symlink())
            self.assertEqual(expanded.read_bytes(), b"fake safetensors payload")


if __name__ == "__main__":
    unittest.main()
