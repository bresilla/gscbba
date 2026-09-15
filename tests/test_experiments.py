from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gscbba import experiments


class ExperimentOutputTests(unittest.TestCase):
    def test_figures_only_uses_selected_output_without_running_experiments(self):
        from gscbba import make_figures

        with tempfile.TemporaryDirectory() as temporary:
            selected = str(Path(temporary) / "selected")
            with (
                patch.object(experiments, "RESULTS_DIR", selected),
                patch.object(experiments, "run_scalability") as run_scalability,
                patch.object(make_figures, "RESULTS", "unused"),
                patch.object(make_figures, "FIGDIR", "unused"),
                patch.object(make_figures, "main") as plot,
            ):
                experiments.main(["--figures-only", "--output-dir", selected])
                run_scalability.assert_not_called()
                plot.assert_called_once_with()
                self.assertEqual(make_figures.RESULTS, selected)
                self.assertEqual(make_figures.FIGDIR, str(Path(selected) / "figures"))

    def test_quick_run_does_not_overwrite_production_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = root / "results"
            production.mkdir()
            cached = production / "scalability.json"
            cached.write_text('{"production": true}\n')
            with (
                patch.object(experiments, "RESULTS_DIR", str(production)),
                patch.object(
                    experiments, "run_scalability", return_value={"trials": 2}
                ),
                patch("sys.argv", ["experiments", "--quick", "--only", "scalability"]),
            ):
                experiments.main()
            self.assertEqual(json.loads(cached.read_text()), {"production": True})
            quick = json.loads(
                (root / "quick-results" / "scalability.json").read_text()
            )
            self.assertEqual(quick["trials"], 2)
            self.assertEqual(quick["metadata"]["schema_version"], 2)
            self.assertTrue(quick["metadata"]["quick"])


if __name__ == "__main__":
    unittest.main()
