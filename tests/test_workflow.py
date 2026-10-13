import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResearchWorkflow(unittest.TestCase):
    def test_committed_birthweight_smoke_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt"
            environment = os.environ.copy()
            environment["RESEARCH_MAX_ROWS"] = "2000"
            completed = subprocess.run(
                [sys.executable, "run_research.py", "econometrics_III.ipynb", "--output", str(output)],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            receipt = json.loads((output / "execution.json").read_text())
            self.assertEqual(receipt["model_results"]["sample_rows"], 2000)
            self.assertEqual(receipt["sample_controls"], {"RESEARCH_MAX_ROWS": "2000"})
            self.assertLess(receipt["model_results"]["selection_index_min"], 0)
            self.assertGreater(receipt["model_results"]["selection_index_max"], 0)
            self.assertLess(receipt["model_results"]["generalized_residual_min"], 0)
            self.assertGreater(receipt["model_results"]["generalized_residual_max"], 0)
            self.assertIn("not a validated causal", receipt["model_results"]["scope"])

    def test_too_small_smoke_sample_fails_before_estimation(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["RESEARCH_MAX_ROWS"] = "1999"
            completed = subprocess.run(
                [sys.executable, "run_research.py", "econometrics_III.ipynb", "--output", directory],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("at least 2000 rows", completed.stdout)


if __name__ == "__main__":
    unittest.main()
