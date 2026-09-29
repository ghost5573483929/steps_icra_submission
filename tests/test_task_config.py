import json
import unittest
from pathlib import Path

from config import Config, scoped_task_config, stable_config_hash


ROOT = Path(__file__).resolve().parents[1]


class TaskConfigTests(unittest.TestCase):
    def test_task_contents_participate_in_run_hash(self):
        cfg = Config()
        task = json.loads((ROOT / "configs/task.json").read_text())
        changed = dict(task, history_steps=task["history_steps"] + 1)
        self.assertNotEqual(
            stable_config_hash({"config": cfg, "task": task}),
            stable_config_hash({"config": cfg, "task": changed}),
        )

    def test_unrelated_dataset_override_does_not_change_hash(self):
        task = json.loads((ROOT / "configs/task.json").read_text())
        changed = json.loads(json.dumps(task)); changed["datasets"]["unrelated"]={"chunk_stride":99}
        self.assertEqual(stable_config_hash(scoped_task_config(task,"eth")),stable_config_hash(scoped_task_config(changed,"eth")))

    def test_missing_task_path_raises_file_not_found(self):
        missing_path = ROOT / "configs" / "does_not_exist_task.json"
        self.assertFalse(missing_path.exists())
        with self.assertRaises(FileNotFoundError):
            Config(task_path=missing_path)

    def test_task_json_without_dense_stride_resolves_unaffected(self):
        cfg_before = Config()
        stride_before = cfg_before.forecasting.window.stride

        task = json.loads((ROOT / "configs/task.json").read_text())
        self.assertNotIn("dense_stride", task.get("forecasting", {}))
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.json"
            task_path.write_text(json.dumps(task))
            cfg_after = Config(task_path=task_path)

        self.assertEqual(cfg_after.forecasting.window.stride, stride_before)

    def test_task_json_with_dense_stride_raises_named_retired_error(self):
        task = json.loads((ROOT / "configs/task.json").read_text())
        task.setdefault("forecasting", {})["dense_stride"] = 5
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.json"
            task_path.write_text(json.dumps(task))
            with self.assertRaises(ValueError) as ctx:
                Config(task_path=task_path)

        message = str(ctx.exception)
        self.assertIn("dense_stride", message)
        self.assertIn("retired", message)
        self.assertNotIn("Unknown forecasting config field", message)

    def test_dense_stride_check_does_not_fire_on_other_forecasting_keys(self):
        task = json.loads((ROOT / "configs/task.json").read_text())
        task.setdefault("forecasting", {})["totally_unrecognised_key"] = 5
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.json"
            task_path.write_text(json.dumps(task))
            with self.assertRaises(ValueError) as ctx:
                Config(task_path=task_path)

        message = str(ctx.exception)
        self.assertIn("Unknown forecasting config field", message)
        self.assertIn("totally_unrecognised_key", message)
        self.assertNotIn("dense_stride", message)


if __name__ == "__main__": unittest.main()
