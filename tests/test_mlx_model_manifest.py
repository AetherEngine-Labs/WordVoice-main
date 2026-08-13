import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlx_wordvoice.model_manifest import validate_model_manifest


class ModelManifestTest(unittest.TestCase):
    def create_model(self, root: Path, *, contract: str = "wordvoice-mlx-model.v3"):
        noise = b"fixed-pytorch-noise"
        (root / "rand_noise.npy").write_bytes(noise)
        manifest = {
            "contract": contract,
            "files": {
                "rand_noise.npy": {
                    "bytes": len(noise),
                    "sha256": hashlib.sha256(noise).hexdigest(),
                }
            },
        }
        (root / "wordvoice.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_accepts_complete_v3_package(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.create_model(root)

            self.assertEqual(validate_model_manifest(root), expected)

    def test_accepts_selective_qwen_v4_package(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.create_model(root, contract="wordvoice-mlx-model.v4")
            expected["quantization"] = {
                "bits": 4,
                "components": ["qwen2.model.layers"],
                "group_size": 64,
                "mode": "affine",
            }
            (root / "wordvoice.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )

            self.assertEqual(validate_model_manifest(root), expected)

    def test_rejects_broader_or_unknown_v4_quantization(self):
        invalid = (
            {"bits": 3, "components": ["qwen2.model.layers"], "group_size": 64},
            {"bits": 4, "components": ["flow"], "group_size": 64},
            {"bits": 4, "components": ["qwen2.model.layers"], "group_size": 32},
        )
        for quantization in invalid:
            with self.subTest(quantization=quantization), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.create_model(root, contract="wordvoice-mlx-model.v4")
                manifest["quantization"] = quantization
                (root / "wordvoice.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

                with self.assertRaisesRegex(ValueError, "quantization contract mismatch"):
                    validate_model_manifest(root)

    def test_rejects_superseded_contracts(self):
        for contract in ("wordvoice-mlx-model.v1", "wordvoice-mlx-model.v2"):
            with self.subTest(contract=contract), TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.create_model(root, contract=contract)

                with self.assertRaisesRegex(ValueError, "unsupported.*contract"):
                    validate_model_manifest(root)

    def test_rejects_missing_or_modified_noise(self):
        for defect in ("metadata", "file", "checksum"):
            with self.subTest(defect=defect), TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.create_model(root)
                if defect == "metadata":
                    manifest["files"].pop("rand_noise.npy")
                    (root / "wordvoice.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                elif defect == "file":
                    (root / "rand_noise.npy").unlink()
                else:
                    manifest["files"]["rand_noise.npy"]["sha256"] = "0" * 64
                    (root / "wordvoice.json").write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )

                with self.assertRaises((ValueError, FileNotFoundError)):
                    validate_model_manifest(root)


if __name__ == "__main__":
    unittest.main()
